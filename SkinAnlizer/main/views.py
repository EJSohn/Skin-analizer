from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.template import RequestContext
from django.core.urlresolvers import reverse
from django.shortcuts import render_to_response

from forms import UploadFileForm
from models import UploadFile

# Create your views here.

def index(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            new_file = UploadFile(file = request.FILES('file'))
            new_file.save()
            return HttpResponseRedirect(reverse('main:index'))
    else :
        form = UploadFileForm()

    data = {'form': form}
    return render(request, 'main/index.html', data, content_type='application/xhtml+xml')

