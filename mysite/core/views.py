from django.shortcuts import render, redirect, render_to_response
from django.template import RequestContext
from django.views.generic import TemplateView, ListView, CreateView
from django.core.files.storage import FileSystemStorage
from django.urls import reverse_lazy
from .SBNC.SBNC import SBNC

from .forms import FileForm
from .models import File


class Home(TemplateView):
    template_name = 'home.html'

def notFound(request, exception, template_name="error404.html"):
    response = render_to_response(template_name)
    #response.status_code = 404
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
    if request.method == 'POST':
        file = File.objects.get(pk=pk)
        SBNC(file.file,file.temporalOrder.file,file.posColumn,file.negColumn)
    return redirect('home')
