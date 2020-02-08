#################################################################################
# Author: Blai Ras                                                               #
# Bachelor Thesis developed with Eurecat and Universitat of Barcelona            #
# January 2020                                                                   #
# Title: Detecting discrimination through Suppes Bayes Causal Network            #
# Based on the work: https://link.springer.com/article/10.1007/s41060-016-0040-z #
#################################################################################

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path

from mysite.core import views

urlpatterns = [
    path('', views.Home.as_view(), name='home'),
    path('files/', views.file_list, name='file_list'),
    path('files/upload/', views.upload_file, name='upload_file'),
    path('files/<int:pk>/', views.delete_file, name='delete_file'),
    path('files/anlysis/<int:pk>/', views.categorize, name='categorize'),
    path('files/anlysis/<int:pk>', views.start_disc, name='start_disc'),
    path('PageRankScore/<slug:name>', views.pageRankExam, name='pageRankShow'),
    path('scores/saveTable/', views.saveTable, name=''),

    path('admin/', admin.site.urls),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# Handlers for custom error templates!
handler404 = 'mysite.core.views.notFound'
handler500 = 'mysite.core.views.notFound500'
