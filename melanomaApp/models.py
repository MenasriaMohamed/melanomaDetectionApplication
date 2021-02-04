from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils.safestring import mark_safe

class Doctor(models.Model):
    '''
        the Doctor model
    '''
    firstName = models.CharField(max_length=30, null=False, blank=False ,default="Doctor")
    lastName = models.CharField(max_length=30, null=False, blank=False ,default="Doctor")    
    phone = models.CharField(max_length=15, null=True, blank=True)
    image = models.ImageField(upload_to='avatars', null=True, blank=True, default="avatars/av.png")
    user = models.OneToOneField(User, on_delete=models.CASCADE, null=True, blank=True)
    class Meta:
        verbose_name = 'Doctor'
    def __str__(self):
        if self.user:
            return self.user.email
        else:
            return 'h'

class Patient(models.Model):
    '''
        the Patient model
    '''
    firstName = models.CharField(max_length=30, null=False, blank=False)
    lastName = models.CharField(max_length=30, null=False, blank=False)
    birthDate = models.DateTimeField(null=True, blank=True)     
    address = models.TextField(null=True, blank=True)
    phone = models.CharField(max_length=15, null=True, blank=True)
    email = models.EmailField(max_length=254,null=True, blank=True)
    sexe = models.CharField(max_length=30, null=False, blank=False, default="male")
    dateCreation = models.DateTimeField('Creation date', auto_now_add=True)
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE, null=True, blank=True, related_name='patients')
    
    def __str__(self):
        return self.firstName+' '+self.lastName+' ('+self.phone+')'

class Image(models.Model):
    '''
        the Image model
    '''
    name = models.CharField(max_length=30)
    image = models.ImageField(upload_to='images', default=None)
    date = models.DateTimeField('upload date', auto_now_add=True)
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, null=True, blank=True)
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE, null=True, blank=True, related_name='images')
    result = models.IntegerField('result', default=1)
    type = models.CharField(max_length=30, default='PH2')
    method = models.IntegerField('mothde abcd 7pcl menz', default=4)
    def __str__(self):
        return self.name+' '+str(self.date)+' '+self.image.url

class Note(models.Model):
    '''
        Note model
    '''
    title = models.CharField(max_length=254, null=True, blank=True)
    content = models.TextField(null=True, blank=True)
    date = models.DateTimeField('Note date', auto_now_add=True)
    image = models.ForeignKey(Image, on_delete=models.CASCADE, null=True, blank=True)
    
    def __str__(self):
        return self.title +' '+self.content

class Details(models.Model):
    '''
        the image Details model
    '''
    image = models.OneToOneField(Image, on_delete=models.CASCADE, null=True, blank=True)
    
    ###### preparation 
    preprocess = models.ImageField(upload_to='images', null=True, blank=True)    
    segmentation = models.ImageField(upload_to='images', null=True, blank=True)   
    posttraitement = models.ImageField(upload_to='images', null=True, blank=True)

    ###### asymmetry
    homologue = models.ImageField(upload_to='images', null=True, blank=True)
    subregion = models.ImageField(upload_to='images', null=True, blank=True)
    distance = models.ImageField(upload_to='images', null=True, blank=True)

    ###### border
    border = models.ImageField(upload_to='images', null=True, blank=True)
    borderlength = models.ImageField(upload_to='images', null=True, blank=True)
    
    ######## diametre 
    enclosingCircle = models.ImageField(upload_to='images', null=True, blank=True)    
    openCircle = models.ImageField(upload_to='images', null=True, blank=True)   
    lengtheningIndex = models.ImageField(upload_to='images', null=True, blank=True)

    ######## color
    kmeans = models.ImageField(upload_to='images', null=True, blank=True)
    kmeans2 = models.ImageField(upload_to='images', null=True, blank=True)
    hsv = models.ImageField(upload_to='images', null=True, blank=True)
    yuv = models.ImageField(upload_to='images', null=True, blank=True)
    ycbcr = models.ImageField(upload_to='images', null=True, blank=True)

    extract = models.ImageField(upload_to='images', null=True, blank=True)
    contour = models.ImageField(upload_to='images', null=True, blank=True)
    circle = models.ImageField(upload_to='images', null=True, blank=True)
    rect = models.ImageField(upload_to='images', null=True, blank=True)

    def __str__(self):
        return str(self.image)

class Caracteristic(models.Model):
    '''
        Image Caracteristic model
    '''
    car0 = models.FloatField('asymmetryByBestFitEllipse', default=0)
    car1 = models.FloatField('asymmetryByDistanceByCircle', default=0)
    car2 = models.FloatField('asymmetryIndex', default=0)
    car3 = models.FloatField('asymmetryBySubRegion', default=0)
    car4 = models.FloatField('asymmetryBySubRegionCentered', default=0)
    car5 = models.FloatField('asymmetryBySubRegionCentered2', default=0)
    car6 = models.FloatField('borderRoundness', default=0)
    car7 = models.FloatField('borderLength', default=0)
    car8 = models.FloatField('borderRegularityIndex', default=0)
    car9 = models.FloatField('borderRegularityIndexRatio', default=0)
    car10 = models.FloatField('borderCompactIndex', default=0)
    car11 = models.FloatField('borderHeywoodCircularityIndex', default=0)
    car12 = models.FloatField('borderHarrisCorner', default=0)
    car13 = models.FloatField('borderFractalDimension', default=0)
    car14 = models.FloatField('colorHSVIntervals', default=0)
    car15 = models.FloatField('colorYUVIntervals', default=0)
    car16 = models.FloatField('colorYCbCrIntervals', default=0)
    car17 = models.FloatField('colorSDG', default=0)
    car18 = models.FloatField('colorKurtosis', default=0)
    car19 = models.FloatField('diameterMinEnclosingCircle', default=0)
    car20 = models.FloatField('diameterOpenCircle', default=0)
    car21 = models.FloatField('diameterLengtheningIndex', default=0)
    car22 = models.FloatField('inflammationAndBloodness', default=0)
    car23 = models.FloatField('sensibility', default=0)
    car24 = models.FloatField('darkPoints', default=0)
    car25 = models.FloatField('blueGrey', default=0)
    date = models.DateTimeField('computing date', auto_now_add=True)
    image = models.OneToOneField(Image, on_delete=models.CASCADE, null=True, blank=True)

    def __str__(self):
        return str([self.car0,self.car1,self.car2,self.car3,self.car4,self.car5,self.car6,self.car7,self.car8,self.car9,self.car10,
        self.car11,self.car12,self.car13,self.car14,self.car15,self.car16,self.car17,self.car18,self.car19,self.car20,self.car21, self.car22, self.car23, self.car24, self.car25])