from django.urls import path
from .views import *

urlpatterns = [

    path('users/', UserList.as_view()),
    path('users/<int:pk>/', UserDetail.as_view()),
    path('posts/', PostList.as_view()),
    path("posts/", PostList.as_view(), name="post-list"),
    path("posts/<slug:slug>/", PostDetail.as_view(), name="post-detail"),
    path('posts/<int:pk>/', PostDetail.as_view()),
    path('categories/', CategoryList.as_view()),
    path('categories/<int:pk>/', CategoryDetail.as_view()),
    path('tags/', TagList.as_view()),
    path('tags/<int:pk>/', TagDetail.as_view()),
    path('comments/', CommentList.as_view()),
    path('comments/<int:pk>/', CommentDetail.as_view()),
    path('contacts/', ContactList.as_view()),
    path('contacts/<int:pk>/', ContactDetail.as_view()),
]
