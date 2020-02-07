#################################################################################
# Author: Blai Ras                                                               #
# Bachelor Thesis developed with Eurecat and Universitat of Barcelona            #
# January 2020                                                                   #
# Title: Detecting discrimination through Suppes Bayes Causal Network            #
# Based on the work: https://link.springer.com/article/10.1007/s41060-016-0040-z #
#################################################################################

from django import forms

from .models import File


class FileForm(forms.ModelForm):
    class Meta:
        model = File
        fields = ('file', 'decColumn', 'temporalOrder')
