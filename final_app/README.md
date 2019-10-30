# Простой поисковик
Поисковик, умеющий индексировать коллекцию файлов (построчно) и запускать сайт с выдачей. Включает модели TF-IDF, BM25, FastText, ELMo и BERT. Для работы необязательны GPU, но крайне желателен большой объём RAM.

# Установка
Окружение проще всего устанавливается с помощью волшебной `conda-forge`, которая умеет решать конфликты между версиями пакетов и устанавливать битые пакеты (проверено на `ufal.udpipe`, `tensorflow`, `uwsgi` и `mysql-client`). Код ниже, выполненный из склонированного репозитория, создаст виртуальное окружение `infosearch` и установит в него нужные пакеты:
```
conda create -y --name infosearch python==3.6.8
source activate infosearch
conda install -f -y -q --name infosearch -c conda-forge --file conda_requirements.txt
pip install -r requirements.txt
```

Для работы с `jupyter notebook` внутри разных окружений не забудьте установить `conda install -n infosearch nb_conda_kernels`.

Для ускорения работы сложных моделей можно установить `tensorflow-gpu==1.13.1`. Примерный список команд ниже (но он не проверялся):
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.5.0.56/prod/10.0_20190219/cudnn-10.0-linux-ppc64le-v7.5.0.56.tgz
sudo cp -P cuda/targets/ppc64le-linux/include/cudnn.h /usr/local/cuda-10.0/include/
sudo cp -P cuda/targets/ppc64le-linux/lib/libcudnn* /usr/local/cuda-10.0/lib64/
sudo chmod a+r /usr/local/cuda-10.0/lib64/libcudnn*
```

# Индексация и запуск сайта
В код входят модули для работы веб-версии, которые интегрированы с индексацией. Для запуска индексации по умолчанию нужно положить [файл](https://www.kaggle.com/loopdigga/quora-question-pairs-russian) отсюда в папку ./index, модель [FastText](http://vectors.nlpl.eu/repository/11/181.zip) распаковать в папку ./models/fasttext, модель [ELMo](http://vectors.nlpl.eu/repository/11/196.zip) положить в папку ./models/elmo и модель [BERT](http://files.deeppavlov.ai/deeppavlov_data/bert/rubert_cased_L-12_H-768_A-12_v2.tar.gz) положить в папку ./models/rubert. Для работы программа должна связаться с локальным сервером bert-serving-server, которую можно запустить параллельным процессом, например, так:
```
tmux
source activate infosearch
bert-serving-start -model_dir /home/isikus/final_app/models/rubert -num_worker=1 -max_seq_len=40
<Ctrl+Z>
bg
tmux detach
```
Далее нужно запустить сервер: `python main.py`, он автоматически проиндексирует файл и сохранит все модели. После индексации для корректной работы ELMo сервер нужно перезапустить: `flask run --host 0.0.0.0 --port 5000 --without-threads`.

# Спасибо за предмет:)
