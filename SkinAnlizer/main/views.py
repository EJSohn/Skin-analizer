from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.template import RequestContext
from django.core.urlresolvers import reverse
from django.shortcuts import render_to_response

from forms import UploadFileForm
from models import UploadFile
from sparse_test import run_test

# Create your views here.

def index(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            new_file = UploadFile(file = request.FILES['file'])
            new_file.save()
            r_pred = run_test()
            data = {'r_pred':r_pred}
            print r_pred
            #return render(request, 'main/index2.html', data)
    else :
        form = UploadFileForm()

    data = {'form': form, 'r_pred':'UnKnown'}

    return render(request, 'main/index.html', data)

