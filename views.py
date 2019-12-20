from django.shortcuts import render
from uploader.models import UploadForm,Upload
from django.http import HttpResponseRedirect
from django.urls import reverse
# Create your views here.
def home(request):
	if request.method=="POST":
		img = UploadForm(request.POST, request.FILES)
		if img.is_valid():
			img.save()
			return HttpResponseRedirect(reverse('imageupload'))
	else:
		img=UploadForm()
	images=Upload.objects.all().order_by('-upload_date')
	return render(request,'home.html',{'form':img,'images':images})