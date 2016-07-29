from django.db import models


class Original(models.Model):
    hard_text = models.TextField()
    
    def __str__(self):
        return self.hard_text

class Simple(models.Model):
    hard = models.ForeignKey(Original, on_delete=models.CASCADE)
    simple_text = models.TextField()

    def __str__(self):
        return self.simple_text

class Vote(models.Model):
    hard = models.ForeignKey(Original, on_delete=models.CASCADE)
    simple = models.ForeignKey(Simple, on_delete=models.CASCADE)
    votes = models.IntegerField(default=0)

class Simplify(models.Model):
    input = models.TextField()
    output = models.TextField()

class Comment(models.Model):
    history = models.ForeignKey(Simplify, on_delete=models.CASCADE)
    comment = models.TextField()
    
    def __str__(self):
        return self.comment
