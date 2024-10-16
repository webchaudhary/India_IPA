"""ipa_india URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls import include
from django.conf import settings
from django.conf.urls.static import static
from webapp import views as wview 

urlpatterns = [
    path("admin/", admin.site.urls),
    path('accounts/', include('django.contrib.auth.urls')),
    path("", wview.showmap),
    path("addfeature", wview.addFeature),
    path("addlayer", wview.addLayer),
    path("getareas", wview.getAreas),
    path("getreport", wview.getReport),
   path("get-task-status/<str:task_id>/", wview.get_task_status),
    path('getarea-geometry/<int:area_id>/', wview.get_area_geometry, name='get_area_geometry'),
    path('get-command-geometry/<int:command_id>/', wview.get_command_geometry, name='get_command_geometry'),

    path("gettasks", wview.getTasks),
    path("deletetaskhistory/<int:idd>", wview.deleteTaskHistory),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL,
                          document_root=settings.STATIC_ROOT)