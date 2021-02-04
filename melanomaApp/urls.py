from django.urls import path
from django.conf.urls.static import static
from application import settings
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('uploadImg', views.uploadImg, name='uploadImg'),
    path('generate/<int:imgId>', views.generate, name='generate'),
    path('results/<int:imgId>', views.results, name='results'),
    path('images', views.images, name='images'),
    path('images/patient/<int:patientId>', views.patientImages, name='patientImages'),
    path('login', views.login, name='login'),
    path('register', views.register, name='register'),
    path('forms.html', views.forms, name='forms'),
    path('addPatient', views.addPatient, name='addPatient'),
    path('deletePatient/<int:patientId>', views.deletePatient, name='deletePatient'),
    path('updatePatient/<int:patientId>', views.updatePatient, name='updatePatient'),
    path('patientsList', views.patientsList, name='patientsList'),
    path('preparation/<int:imgId>', views.preparation, name='preparation'),
    path('asymmetry', views.asymmetry, name='asymmetry'),
    path('border', views.border, name='border'),
    path('color', views.color, name='color'),
    path('diameter', views.diameter, name='diameter'),
    path('notesList/<int:imgId>', views.notesList, name='notesList'),
    path('addNote/<int:imgId>', views.addNote, name='addNote'),
    path('deleteNote/<int:noteId>', views.deleteNote, name='deleteNote'),
    path('dashboard', views.dashboard, name='dashboard'),
    path('settings', views.settings, name='settings'),
    path('logout', views.logout, name='logout'),
    path('user', views.user, name='user'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
