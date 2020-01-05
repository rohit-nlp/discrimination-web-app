
import pandas as pd
import seaborn as sns

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
        reason,df,probs,scores,disconnectedNodes,pos,neg,neut,elapsed = SBNC(file.file,file.temporalOrder.file,file.posColumn,file.negColumn)
        if scores is not None:

            request.session['df'] = df.to_json(orient='split')
            request.session['probs'] = probs.to_json()
            request.session['columns'] = pd.DataFrame({'pos':file.posColumn,'neg':file.negColumn},index=[0]).to_json()

            pd.set_option('display.max_colwidth', -1)
            scores['Name'] = scores['Name'].apply(lambda x: '<a href="http://127.0.0.1:8000/PageRankScore/{0}">{0}</a>'.format(x))
            return render(request,"results.html",{'reason':reason,'scores':scores.to_html(
                classes="table table-striped table-bordered table-sm",
                table_id="scoreTable",
                index=False,
                escape=False,
                justify='initial'),'pos':pos,'neg':neg,'neut':neut,
              'disconnected':disconnectedNodes.to_html(
                  classes="table table-borderless table-hover table-striped table-sm",
                  index=False,
                  justify="center"
              ),'elapsed':elapsed})
    return render(request, "results.html", {'file': file, 'reason': reason, 'probs': probs, 'scores': scores})
    #return render(request, "results.html")

def pageRankExam(request,name):

    df = pd.read_json(request.session.get('df'),orient='split')
    probs = pd.read_json(request.session.get('probs'))
    columns = pd.read_json(request.session.get('columns'))
    print("start pr")
    PRScores = pageRank(df,probs,columns['pos'][0],columns['neg'][0])
    PRScores.to_csv("PRinWeb.csv",index=None,sep=";")
    print("done Pr")
    if PRScores is not None:
        # sns.set_context("talk")
        sns.set()
        sns.despine()
        # Create an array with the colors you want to use
        colors = ["#E3D4AD", "#ffb39c"]
        #sns.set_palette(sns.color_palette(colors))
        sns_plot = sns.lmplot(height=6,
                              y='Negative Discrimination', x='Positive Discrimination', data=PRScores,
                              hue=name, fit_reg=False)
        print("start p1")
        sns_plot.savefig("media/smallPlot.png", dpi=200)
        print("start p2")
        sns_plot = sns.lmplot(height=10,
                              y='Negative Discrimination', x='Positive Discrimination', data=PRScores,
                              hue=name, fit_reg=False)
        sns_plot.savefig("media/bigPlot.png", dpi=200)
        print("end p2")

        return render(request,"pageRankShow.html",{'reason':"",'name':name})
    return render(request, "pageRankShow.html", {'reason': "PageRank Scores could not be computed"})

