from django.contrib import admin

from .models import Vote
from .models import Simple
from .models import Original
from .models import Simplify
from .models import Comment

admin.site.register(Vote)
admin.site.register(Simple)
admin.site.register(Original)
admin.site.register(Simplify)
admin.site.register(Comment)
