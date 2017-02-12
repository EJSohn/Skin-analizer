from __future__ import unicode_literals

from django.db import models

# Create your models here.

class UploadFile(models.Model):
    file = models.FileField(upload_to='files/')