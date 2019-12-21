from django.db import models


class File(models.Model):

    file = models.FileField()

    def __str__(self):
        return self.title

    def delete(self, *args, **kwargs):
        self.file.delete()
        super().delete(*args, **kwargs)
