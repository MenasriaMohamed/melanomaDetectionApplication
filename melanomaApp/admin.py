from django.contrib import admin

# Register your models here.
from .models import Doctor
admin.site.register(Doctor)

from .models import Image
admin.site.register(Image)

from .models import Caracteristic
admin.site.register(Caracteristic)

from .models import Patient
admin.site.register(Patient)

from .models import Details
admin.site.register(Details)