from django.shortcuts import render

# Create your views here.
def index_view(request):
    return render(request, "detect/index.html")

def about_view(request):
    return render(request, "detect/about.html")

def test_view(request):
    return render(request, "detect/test.html")
