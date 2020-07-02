from django import forms
from django.contrib.auth.forms import UserChangeForm, AuthenticationForm
from django.contrib.auth import get_user_model
from .models import Profile


class CustomUserChangeForm(UserChangeForm):
    class Meta:
        model = get_user_model() #=> auth.User
        fields = ('username', 'email',
                'first_name', 'last_name',)


class ProfileForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ('nickname', 'image', )


class CustomAuthenticationForm(AuthenticationForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        class_update_fields = ('username', 'password', )
        for field_name in class_update_fields:
            self.fields[field_name].widget.attrs.update({
                'class': 'form-control'
            })
