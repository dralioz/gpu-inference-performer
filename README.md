# YOLO26n FastAPI Inference

Ultralytics YOLO26 nano modeli ile görsel inference yapan minimal bir FastAPI servisi.

## Kurulum

Conda ile ortamı oluşturmak için:

```bash
conda env create -f conda.yaml
conda activate gpu-inference-performer
```

macOS üzerinde proje klasöründe:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

İlk çalıştırmada Ultralytics, `yolo26n.pt` modelini indirir. Model adı kullanıcı isteğindeki `yolov26n` ifadesine karşılık Ultralytics formatında `yolo26n.pt` olarak kullanılır.

## Çalıştırma

```bash
uvicorn app.main:app --reload
```

API dokümantasyonu için `http://127.0.0.1:8000/docs` adresini açın.

## Locust ile load test

Servis çalışırken başka bir terminalde Locust'u başlatın:

```bash
locust -f locustfile.py --host http://127.0.0.1:8000
```

Ardından `http://localhost:8089` adresini açıp kullanıcı sayısını ve artış hızını
belirleyin. `locustfile.py` her kullanıcı için test amaçlı bir JPEG oluşturur ve
`POST /predict` endpoint'ine multipart dosya olarak gönderir.

Komut satırından arayüzsüz test örneği:

```bash
locust -f locustfile.py --host http://127.0.0.1:8000 \
  --headless -u 10 -r 2 -t 1m
```

## Endpoint'ler

- `GET /health`: Servis ve model durumunu döndürür.
- `POST /predict`: `file` alanı ile bir görsel alır ve tespitleri JSON olarak döndürür.

Örnek istek:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg"
```

## Ayarlar

Varsayılan cihaz CPU'dur. Farklı bir cihaz seçmek için:

```bash
DEVICE=mps uvicorn app.main:app --reload
```

Model yolunu değiştirmek için `MODEL_PATH=/path/to/model.pt` kullanabilirsiniz.

Inference çağrısı `ThreadPoolExecutor(max_workers=4)` ile event loop dışına alınır ve
`torch.inference_mode()` içinde çalıştırılır. Böylece aynı servis birden fazla isteği
FastAPI event loop'unu bloklamadan işleyebilir.
