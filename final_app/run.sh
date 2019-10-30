bert-serving-start -model_dir /home/isikus/final_app/models/rubert -num_worker=1 -max_seq_len=40 &
sleep 2m; python app.py
