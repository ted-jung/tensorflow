FROM python:3.12.8

ARG VERSION

LABEL org.label-scheme.version=${VERSION}

# Install the Java Runtime Environment (JRE)
# The 'default-jre' meta-package is used for better compatibility.
# 'ca-certificates-java' is also added for proper SSL/TLS support.
RUN apt-get update && \
    apt-get install -y default-jre ca-certificates-java && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the JAVA_HOME environment variable
# This helps Konlpy find the JVM library.
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

RUN python3 -m pip install --upgrade pip

COPY ./movie_review/requirements.txt /ted/requirements.txt

COPY ./movie_review/ted_naver_movie_sgd_model.joblib /ted/ted_naver_movie_sgd_model.joblib

COPY ./movie_review/ted-test.py /ted

WORKDIR /ted

RUN pip install -r requirements.txt

ENTRYPOINT [ "python3" ]

CMD [ "ted-test.py" ]