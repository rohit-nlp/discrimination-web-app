
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import time

from django.shortcuts import render, redirect, render_to_response
from django.views.generic import TemplateView
from django.core import serializers

from .SBNC.SBNC import SBNC
from .forms import FileForm
from .models import File
from .SBNC.ComputeDiscrimination import pageRank


class Home(TemplateView):
    template_name = 'home.html'

def notFound(request,exception):
    response = render_to_response("error404.html")
    response.status_code = 404
    return response

def file_list(request):
    files = File.objects.all()
    return render(request, 'file_list.html', {
        'files': files
    })


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


def delete_file(request, pk):
    if request.method == 'POST':
        file = File.objects.get(pk=pk)
        file.delete()
    return redirect('file_list')

def start_disc(request,pk):
    if request.method == 'GET':
        file = File.objects.get(pk=pk)
        reason,df,invalidMarginal, notDistinguish,probs,scores,disconnectedNodes,pos,neg,neut,explainable,inco,apparent,elapsed = SBNC(file.file,file.temporalOrder.file,file.posColumn,file.negColumn)
        if scores is not None:
            eventInfo = ""
            if invalidMarginal:
                eventInfo = "Following events were deleted because an invalid marginal probability: " + ', '.join(invalidMarginal) + " ."
            print(notDistinguish)
            if notDistinguish and invalidMarginal:
                eventInfo = eventInfo + "  And "+ ', '.join(notDistinguish)
            elif notDistinguish:
                eventInfo = "  And "+ ', '.join(notDistinguish)

            request.session['df'] = df.to_json(orient='split')
            request.session['probs'] = probs.to_json(orient='split')
            request.session['columns'] = pd.DataFrame({'pos':file.posColumn,'neg':file.negColumn},index=[0]).to_json(orient='split')

            pd.set_option('display.max_colwidth', -1)
            #scores['Name'] = scores['Name'].apply(lambda x: '<button type="button" class="btn btn-light waves-effect btn-sm">{0}</button>'.format(x))
            scores['Name'] = scores['Name'].apply(lambda x: '<u><a style="color:#0000EE;" href="/PageRankScore/{0}">{0}</a></u>'.format(x))

            return render(request,"results.html",{'reason':reason,'scores':scores.to_html(
                classes="table table-striped table-bordered table-sm w-auto",
                table_id="scoreTable",
                index=False,
                escape=False,
                justify='left'),'pos':pos,'neg':neg,'neut':neut,'explainable':explainable,'inco':inco,'apparent':apparent,
              'disconnected':disconnectedNodes.to_html(
                  classes="table table-borderless table-striped table-sm",
                  table_id="disconnectedTable",
                  index=False,
                  justify="center"
              ),'elapsed':elapsed, 'eventInfo':eventInfo})
    return render(request, "results.html", {'file': file, 'reason': reason, 'probs': probs, 'scores': scores})
    #return render(request, "results.html")

def pageRankExam(request,name):
    elapsed = time.time()
    df = pd.read_json(request.session.get('df'),orient='split')
    probs = pd.read_json(request.session.get('probs'),orient='split')
    columns = pd.read_json(request.session.get('columns'),orient='split')
    reason = "PageRank Scores could not be computed"
    if probs is not None and df is not None and columns is not None:
        if name in df.columns:
            print("start pr")
            PRScores = pageRank(df,probs,columns['pos'][0],columns['neg'][0],name)
            print("done Pr")
            PRScores.to_csv("PRscores.csv",index=None,sep=";")
            if PRScores is not None:
                createGraphs(PRScores,name)
                elapsed = time.strftime('%H:%M:%S', time.gmtime((time.time() - elapsed)))
                return render(request,"pageRankShow.html",{'reason':"",'name':name,'elapsed':elapsed})
        else:
            reason = "This variable is not present in the dataset"
    return render(request, "pageRankShow.html", {'reason': reason})

def createGraphs(PRScores,name):
    sns.set()
    sns.despine()
    # Create an array with the colors you want to use
    colors = ["#5bc0de", "#d9534f"]
    sns.set_palette(sns.color_palette(colors))

    fig, axs = plt.subplots(figsize=(15, 15))
    sns_plot = sns.lmplot(y='Negative Discrimination', x='Positive Discrimination', data=PRScores,
                          hue=name, fit_reg=False)
    plt.savefig("media/smallPoints.png", dpi=200)
    plt.close()



    fig, axs = plt.subplots(ncols=2, figsize=(20, 10))
    sns.distplot(PRScores[PRScores[name] == 1]['Positive Discrimination'], label = name + ": 1",color = "#d9534f",hist=False, ax=axs[0])
    sns.distplot(PRScores[PRScores[name] == 0]['Positive Discrimination'], label = name + ": 0",color = "#5bc0de",hist=False, ax=axs[0])
    sns.distplot(PRScores[PRScores[name] == 1]['Negative Discrimination'], label = name + ": 1",color = "#d9534f",hist=False,
                 ax=axs[1])
    sns.distplot(PRScores[PRScores[name] == 0]['Negative Discrimination'], label = name + ": 0",color = "#5bc0de",hist=False,
                 ax=axs[1])

    plt.savefig('media/distplot.png',dpi=200)

    # fig, axs = plt.subplots(figsize=(20, 10))
    #
    # sns.distplot(PRScores[PRScores[name] == 1]['Negative Discrimination'], label=name + ": 1", color="#d9534f",
    #              hist=False,
    #              )
    # sns.distplot(PRScores[PRScores[name] == 0]['Negative Discrimination'], label=name + ": 0", color="#5bc0de",
    #              hist=False,
    #              )
    # plt.savefig('media/temp.png')

    fig, axs = plt.subplots(ncols=2, figsize=(20, 15), sharey=True)
    sns.boxplot(x=name, y="Positive Discrimination", data=PRScores,
                boxprops={'facecolor': '#5bc0de'}, showcaps=False, showfliers=False, ax=axs[0])
    sns.boxplot(x=name, y="Negative Discrimination", data=PRScores,
                showcaps=False, boxprops={'facecolor': '#d9534f'},
                showfliers=False, ax=axs[1])

    plt.savefig('media/BoxPlot.png', dpi=200)


