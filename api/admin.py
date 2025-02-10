from django.contrib import admin
from .models import *

admin.site.register(Post)
admin.site.register(Category)
admin.site.register(Comment)
# admin.site.register(Contact)
admin.site.register(User)
admin.site.register(Tag)



@admin.register(Contact)
class ContactMessageAdmin(admin.ModelAdmin):
    list_display = ('name', 'email', 'created_at')
    search_fields = ('name', 'email', 'message')
    list_filter = ('created_at',)
