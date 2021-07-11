from django.urls import path
from . import views
urlpatterns = [
    path("",views.index_view,name="index"),
    path("about/",views.about_view, name="about"),
    path("test/",views.test_view, name="test"),
    path(r'^/(?P<stream_path>(.*?))/$',views.dynamic_stream, name="videostream"),
]
