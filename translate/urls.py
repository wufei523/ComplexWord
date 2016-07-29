from django.conf.urls import *
from django.conf.urls import url
from django.contrib import admin

from . import views

urlpatterns = [
                url(r'^$', views.index, name='index'),
                url(r'^result/(?P<hard_text>[\w\W]*)/$', views.result, name='result'),
                url(r'^contribute/', views.contribute, name='contribute'),
                url(r'^add_simp/(?P<pk>\d+)/$', views.add_simp, name='add_simp'),
                url(r'^edit_simp/(?P<pk>\d+)/$', views.edit_simp, name='edit_simp'),
                url(r'^add_origin/', views.add_origin, name='add_origin'),
               
               ]
