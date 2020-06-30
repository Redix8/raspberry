from django.shortcuts import render, redirect
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login as auth_login,  logout as auth_logout

def login(request):
    # 만약, 로그인이 되었다면 index로 돌려 보내기
    if request.user.is_authenticated:
        return redirect('monitor:index')

    if request.method == 'POST':
        # User 검증 + 로그인
        # 1. POST로 넘어온 데이터 form에 넣기
        form = AuthenticationForm(request, request.POST)
        # 2. form 검증 (아이디, 비밀번호 맞음?)
        if form.is_valid():
            # 3. 맞으면, 로그인 시켜줌
            user = form.get_user()
            auth_login(request, user)
            # 4. 로그인 결과 확인이 가능한 페이지로 안내
            return redirect(request.GET.get('next') or 'monitor:index')
            # Quiz. 단축 평가
            # and -> False and True #=> False
            # or -> True or False #=> True
            # a = '' or 'apple' #=> apple
            # b = 'banana' or '' #=> banana

    else:
        # User 로그인 창 보여주기
        form = AuthenticationForm()
    context = {
        'form': form,
    }
    return render(request, 'accounts/login.html', context)