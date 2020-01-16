#################################################################################
#Author: Blai Ras                                                               #
#Bachelor Thesis developed with Eurecat and Universitat of Barcelona            #
#January 2020                                                                   #
#Title: Detecting discrimination through Suppes Bayes Causal Network            #
#Based on the work: https://link.springer.com/article/10.1007/s41060-016-0040-z #
#################################################################################
"""
WSGI config for mysite project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/2.1/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')

application = get_wsgi_application()
