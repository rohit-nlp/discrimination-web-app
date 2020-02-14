#################################################################################
# Author: Blai Ras                                                               #
# Bachelor Thesis developed with Eurecat and Universitat of Barcelona            #
# January 2020                                                                   #
# Title: Detecting discrimination through Suppes Bayes Causal Network            #
# Based on the work: https://link.springer.com/article/10.1007/s41060-016-0040-z #
#################################################################################

import os
import time
from wsgiref.util import FileWrapper

import pandas as pd
import seaborn as sns
from django.http import HttpResponse
from django.shortcuts import render, redirect, render_to_response
from django.views.generic import TemplateView
from matplotlib import pyplot as plt

from .SBNC.CategorizeDataset import adaptDF
from .SBNC.PageRank import pageRank
from .SBNC.ReadDataframes import read
from .SBNC.SBNC import SBNC
from .forms import FileForm
from .models import File


class Home(TemplateView):
    template_name = 'home.html'


# Handler for error 404 (not found)
def notFound(request, exception):
    response = render_to_response("error404.html")
    response.status_code = 404
    return response


# Handler for error 500 (server)
def notFound500(request):
    response = render_to_response("error500.html")
    response.status_code = 500
    return response


# Handle the file list view
def file_list(request):
    files = File.objects.all()
    return render(request, 'file_list.html', {
        'files': files
    })


# Handle the upload of a new model class (file)
def upload_file(request):
    if request.method == 'POST':
        form = FileForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('file_list')
    else:
        form = FileForm()

    return render(request, 'upload_file.html', {
        'form': form
    })


# Handler for file deleting
def delete_file(request, pk):
    if request.method == 'POST':
        file = File.objects.get(pk=pk)
        file.delete()
    return redirect('file_list')


# Handler for the dataset categorization
def categorize(request, pk):
    error = ""
    if request.method == 'POST':
        # Collect the file
        file = File.objects.get(pk=pk)
        # Gather the files
        df, temporalOrder = read(file.file, file.temporalOrder.file)
        if df is not None and temporalOrder is not None:
            # Categorize DF
            df, posColumn, negColumn, error = adaptDF(df, file.decColumn)
            if error == "":
                # Safe categorized DF
                df.to_csv("mysite/core/static/df/Categorized_" + str(file.file), index=None)
                return render(request, "categorized.html",
                              {'dfView': df.head().to_html(classes="table table-striped table-sm",
                                                           table_id="categorizedTable",
                                                           index=False,
                                                           justify="right"), 'pos': posColumn, 'neg': negColumn,
                               'err': error, 'file': file.pk})
        else:
            error = "Can't read the files"

    return render(request, "categorized.html", {'err': error})


# Function that handles the discrimination analysis
def start_disc(request, pk):
    if request.method == 'GET':
        # Collect the file
        file = File.objects.get(pk=pk)
        route_to_df = "mysite/core/static/df/Categorized_" + str(file.file)
        posColumn = request.GET['positiveDrop']
        negColumn = request.GET['negativeDrop']

        # Try to collect every info needed:
        #   Reason: if its not empty, and error occurred
        #   DF: the dataset after the training
        #   InvalidMarginal: tell the user the deleted columns
        #   notDistinguish: tell the user the undistinguished events
        #   probs: dataframe with the graph edges and weights
        #   scores: discrimination scores and veredict
        #   disconnectedNodes: no explanation needed
        #   pos, neg, neut, explainable, inco, apparent: number of variables classified in this type of discrimination
        #   elapsed: time needed for the execution
        reason, df, invalidMarginal, notDistinguish, probs, scores, disconnectedNodes, pos, neg, neut, explainable, inco, apparent, elapsed = SBNC(
            route_to_df,
            file.file, file.temporalOrder.file, file.decColumn, posColumn, negColumn, 0.55, 0.25)
        # If there are no errors
        scores.to_csv("Scores.csv",index=None)
        if scores is not None:
            eventInfo = ""
            # Prepare to tell the deleted events, if any
            if invalidMarginal:
                eventInfo = "Following events were deleted because they have an invalid marginal probability: " + ', '.join(
                    invalidMarginal) + " ."

            # Prepare to tell the merged events, if any
            if notDistinguish and invalidMarginal:
                eventInfo = eventInfo + "  And " + ', '.join(notDistinguish)
            elif notDistinguish:
                eventInfo = "  And " + ', '.join(notDistinguish)

            # Send the datasets to the session, so they will be avaliable in the Page Rank computation class if the user decides to look at it
            # Also send pos and negative column names
            request.session['df'] = df.to_json(orient='split')
            request.session['probs'] = probs.to_json(orient='split')
            request.session['columns'] = pd.DataFrame({'pos': posColumn, 'neg': negColumn},
                                                      index=[0]).to_json(orient='split')

            # Saving scores for download
            scores.to_csv("media/DiscriminationTable.csv", index=None)

            # So the table doesnt shrink
            pd.set_option('display.max_colwidth', -1)

            # Apply a link to the table 'Name' column, color blue and underscript
            scores['Name'] = scores['Name'].apply(
                lambda x: '<u><a style="color:#0000EE;" href="/PageRankScore/{0}">{0}</a></u>'.format(x))
            # Return:
            #   scores.to_html: method that allows me to define the table Bootstrap + CSS class and more style
            return render(request, "results.html", {'reason': reason, 'scores': scores.to_html(
                classes="table table-striped table-bordered table-sm w-auto",
                table_id="scoreTable",
                index=False,
                escape=False,
                justify='left'), 'pos': pos, 'neg': neg, 'neut': neut, 'explainable': explainable, 'inco': inco,
                                                    'apparent': apparent,
                                                    'disconnected': disconnectedNodes.to_html(
                                                        classes="table table-borderless table-striped table-sm",
                                                        table_id="disconnectedTable",
                                                        index=False,
                                                        justify="center"
                                                    ), 'elapsed': elapsed, 'eventInfo': eventInfo,
                                                    'FileName': file.file})
    return render(request, "results.html", {'file': file, 'reason': reason, 'probs': probs, 'scores': scores})


# Function that computes and handles Page Rank Scores
def pageRankExam(request, name):
    # Start measuring time elapsed
    elapsed = time.time()
    # Get the datasets in session
    df = pd.read_json(request.session.get('df'), orient='split')
    probs = pd.read_json(request.session.get('probs'), orient='split')
    columns = pd.read_json(request.session.get('columns'), orient='split')
    # Prepare to return an error variable if needed
    reason = "PageRank Scores could not be computed"
    if probs is not None and df is not None and columns is not None:
        # Check if the name of the variable present in the URL exists in the dataset!
        if name in df.columns:
            print("Starting Page Rank Scores")
            PRScores = pageRank(df, probs, columns['pos'][0], columns['neg'][0], name)
            print("Page Rank Scores computed")
            if PRScores is not None:
                # Create the plots!
                createGraphs(PRScores, name)
                elapsed = time.strftime('%H:%M:%S', time.gmtime((time.time() - elapsed)))
                return render(request, "pageRankShow.html", {'reason': "", 'name': name, 'elapsed': elapsed})
        else:
            reason = "This variable is not present in the dataset"
    return render(request, "pageRankShow.html", {'reason': reason})


# Function that creates the needed plots for the Page Rank scores
def createGraphs(PRScores, name):
    # SNS styling
    sns.set()
    sns.despine()
    # Create an array with the colors you want to use
    colors = ["#5bc0de", "#d9534f"]
    sns.set_palette(sns.color_palette(colors))

    # Point distribution plots
    fig, axs = plt.subplots(figsize=(15, 15))
    sns_plot = sns.lmplot(y='Negative Discrimination', x='Positive Discrimination', data=PRScores,
                          hue=name, fit_reg=False)
    plt.savefig("mysite/core/static/img/smallPoints.png", dpi=200)
    plt.close()

    # Distribution plots
    fig, axs = plt.subplots(ncols=2, figsize=(20, 10))
    sns.distplot(PRScores[PRScores[name] == 1]['Positive Discrimination'], label=name + ": 1", color="#d9534f",
                 hist=False, ax=axs[0])
    sns.distplot(PRScores[PRScores[name] == 0]['Positive Discrimination'], label=name + ": 0", color="#5bc0de",
                 hist=False, ax=axs[0])
    sns.distplot(PRScores[PRScores[name] == 1]['Negative Discrimination'], label=name + ": 1", color="#d9534f",
                 hist=False,
                 ax=axs[1])
    sns.distplot(PRScores[PRScores[name] == 0]['Negative Discrimination'], label=name + ": 0", color="#5bc0de",
                 hist=False,
                 ax=axs[1])

    plt.savefig('mysite/core/static/img/distplot.png', dpi=200)

    # Boxplots,
    fig, axs = plt.subplots(ncols=2, figsize=(20, 15), sharey=True)
    sns.boxplot(x=name, y="Positive Discrimination", data=PRScores,
                boxprops={'facecolor': '#5bc0de'}, showcaps=False, showfliers=False, ax=axs[0])
    sns.boxplot(x=name, y="Negative Discrimination", data=PRScores,
                showcaps=False, boxprops={'facecolor': '#d9534f'},
                showfliers=False, ax=axs[1])

    plt.savefig('mysite/core/static/img/BoxPlot.png', dpi=200)


# Function that downloads the generated score table
def saveTable(request):
    file_path = "media/DiscriminationTable.csv"
    try:
        wrapper = FileWrapper(open(file_path, 'rb'))
        response = HttpResponse(wrapper, content_type='application/force-download')
        response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
        return response
    except Exception as e:
        return None
