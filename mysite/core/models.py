from django.db import models


class File(models.Model):

    file = models.FileField()
    posColumn = models.CharField(max_length=30)
    negColumn = models.CharField(max_length=30)
    temporalOrder = models.FileField()

    def __str__(self):
        return self.posColumn #?

    def delete(self, *args, **kwargs):
        self.file.delete()
        self.temporalorder.delete()
        super().delete(*args, **kwargs)


