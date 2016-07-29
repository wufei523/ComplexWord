from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import Simple, Original, Vote, Simplify, Comment
from .forms import OriginForm, SimpForm, TestForm, CommentForm, SimplifyForm, VoteForm
from django.core.urlresolvers import reverse
from translate import functions, utils


def index(request):
    print("THIS IS request")
    print(request)
    print("THIS IS request.GET")
    print(request.GET)
    #if request get contains "original_text" redirect
    if 'original_text' in request.GET and request.GET["original_text"]:
        hard_text = request.GET.get("original_text")
        return redirect('result', hard_text)
    else:
        form = TestForm()
        return render(request, 'index.html', {'form' : form})


def result(request, hard_text):
    form = TestForm(initial={'original_text': hard_text})
    formc = CommentForm()

    #simple is simplified sentence
    #process is: original word : top 20 words(not handled, directly from model.most_similar())
    simple, complex_word_list = utils.simplify(hard_text)


    if 'original_text' in request.GET and request.GET["original_text"]:
        hard_text = request.GET.get("original_text")
        return redirect('result', hard_text)

    elif request.method == "POST":
        if 'comment' in request.POST:
            comment = request.POST.get("comment")
            comment_form = CommentForm(request.POST)
            history_form = SimplifyForm()
            if comment_form.is_valid():
                try:
                    history = Simplify.objects.get(input=hard_text, output=simple)
                except Simplify.DoesNotExist:
                    history =history_form.save(commit=False)
                    history.input = hard_text
                    history.output=simple
                    history.save()
                    com = comment_form.save(commit=False)
                    com.history = history
                    com.comment = comment
                    com.save()
                    comments = Comment.objects.filter(history=history.id)
                    return render(request, 'result.html', {'simple': simple, 'form' : form, 'formc':formc, 'com':comments, 'complex_word_list': complex_word_list})
            else:
                try:
                    history = Simplify.objects.get(input=hard_text, output=simple)
                except Simplify.DoesNotExist:
                    return render(request, 'result.html', {'simple': simple, 'form' : form, 'formc':formc, 'complex_word_list': complex_word_list})
                comments = Comment.objects.filter(history=history.id)
                return render(request, 'result.html', {'simple': simple, 'form' : form, 'formc':formc, 'com':comments, 'complex_word_list': complex_word_list})
    else:
        try:
            history = Simplify.objects.get(input=hard_text, output=simple)
        except Simplify.DoesNotExist:
            return render(request, 'result.html', {'simple': simple, 'form' : form, 'formc':formc, 'complex_word_list': complex_word_list})
        comments = Comment.objects.filter(history=history.id)
        return render(request, 'result.html', {'simple': simple, 'form' : form, 'formc':formc, 'com':comments, 'complex_word_list': complex_word_list})


def contribute(request):
    origins = Original.objects.all()
    return render(request, 'contribute.html', {'origins' : origins })


def add_simp(request, pk):
    orig = Original.objects.get(pk=pk)
    form = SimpForm()
    results = Simple.objects.filter(hard=pk)
    if request.method =="POST":
        if 'simple_text' in request.POST and request.POST["simple_text"]:
            simple_text = request.POST.get("simple_text")
            save_form = SimpForm(request.POST)
            if save_form.is_valid():
                simp = save_form.save(commit=False)
                simp.simple_text = simple_text
                simp.hard = orig
                simp.save()
                return render(request, 'add_simp.html', {'results': results, 'orig' : orig, 'form': form })
        elif 'votes' in request.POST and request.POST["votes"]:
            votes = request.POST.getlist("votes")
            for vote in votes:
                vote_form = VoteForm()
                simple=Simple.objects.get(hard=orig, simple_text=vote)
                try:
                    v = Vote.objects.get(hard=orig, simple=simple)
                    v.votes += 1
                    v.save()
                except Vote.DoesNotExist:
                    v = vote_form.save(commit=False)
                    v.hard=orig
                    v.simple=simple
                    v.votes=1
                    v.save()
            vote_result = Vote.objects.get(hard=orig, simple=simple)
            return render(request, 'add_simp.html', {'results': results, 'orig' : orig, 'form': form})
        else:
            return render(request, 'add_simp.html', {'results': results, 'orig' : orig, 'form': form})
    else:
        return render(request, 'add_simp.html', {'results': results, 'orig' : orig, 'form': form})


def edit_simp(request, pk):
    simp = Simple.objects.get(pk=pk)
    form = SimpForm(initial={'simple_text': simp.simple_text})
    new_simple_text = request.POST.get("simple_text")
    if request.method =="POST":
        simp.simple_text = new_simple_text
        simp.save()
        return redirect('add_simp', simp.hard.id )
    else:
        return render(request, 'edit_simp.html', {'form' : form})

def add_origin(request):
    hard_text = request.POST.get("hard_text")
    if request.method =="POST":
        save_form = OriginForm(request.POST)
        if save_form.is_valid():
            orig = save_form.save(commit=False)
            orig.hard_text = hard_text
            orig.save()
            origins = Original.objects.all()
        return redirect('contribute')
    else:
        form = OriginForm()
        return render(request, 'add_origin.html', { 'form':form })

