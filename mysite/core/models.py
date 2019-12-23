from django.db import models
from .validators import validate_file_extension


class File(models.Model):

    file = models.FileField(validators=[validate_file_extension])
    posColumn = models.CharField(max_length=30)
    negColumn = models.CharField(max_length=30)
    temporalOrder = models.FileField(validators=[validate_file_extension])

    def __str__(self):
        return self.posColumn #?

    def delete(self, *args, **kwargs):
        self.file.delete()
        self.temporalOrder.delete()
        super().delete(*args, **kwargs)


