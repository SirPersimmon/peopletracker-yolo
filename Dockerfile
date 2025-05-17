FROM gcr.io/kaggle-gpu-images/python

WORKDIR /app

ENV MODELS_PATH="/app/chekpoints"

COPY checkpoints ./chekpoints

RUN git clone https://github.com/SirPersimmon/peopletracker-yolo.git /tmp/peopletracker-yolo && \
    mv /tmp/peopletracker-yolo/src/peopletracker-yolo . && \
    rm -rf /tmp/peopletracker-yolo

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["/app/peopletracker-yolo/webui.py"]
