FROM python:3.11
# RUN useradd -m -u 1000 user
# USER user
# ENV HOME=/home/user \
#     PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app
COPY --chown=user . $HOME/app
COPY ./requirements.txt ~/app/requirements.txt
COPY ./vectorstore/ ~/app/vectorstore/
RUN pip install -r requirements.txt 
RUN pip install -qU --disable-pip-version-check qdrant-client
COPY . .
CMD ["chainlit", "run", "app.py", "--port", "7860"]