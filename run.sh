#!/bin/bash
if [ -z "$VCAP_APP_PORT" ];
then SERVER_PORT=5000;
else SERVER_PORT="$VCAP_APP_PORT";
fi
echo port is------------------- $SERVER_PORT
#python manage.py syncdb --noinput
python manage.py runserver 0.0.0.0:$SERVER_PORT


#SERVER_PORT="$PORT";
#fi
#echo port is $SERVER_PORT
#python manage.py runserver 0.0.0.0:$SERVER_PORT
