from django.shortcuts import render
from src.query import ask

def index(request):
    # a = ask("Is Kazakhstan secular country?")
    # print(a)
    if request.method == 'POST':
        question = request.POST['question']

        response = ask(question)
        context = {
            'answer': response,
            'is_post_request': True,
        }
        return render(request, 'main/index.html', context)
    else:
        return render(request, 'main/index.html', {'is_post_request': False})
