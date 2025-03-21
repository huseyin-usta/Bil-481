# Bil-481

Bu repo Bil 481 dersinin projesi için oluşturulmuştur, toplantı tutanakları, kod, açıklamalar, gereksinimler ve proje ile ilgili dosyalar bu repoda yer alacaktır.

Not: Python notebookumuzu yüklemedik fakat eğer kodlarımızı çalıştırmak isterseniz yapmanız gerekenler çok basit!
Bu kodu çalıştırmak için yapnamız gerekenler:
1) Bir ipynb dosyası oluşturun.
2) models.py den istediğiniz modeli initialize edin.
3) dataset.py den de yazdığımız datasetin iki instancesini oluşturun, burada Albumentations kütüphanesini kullanarak bir transformu parametre olarak kullanabilirsiniz. Bunlardan biri validation yapmak için, diğeri ise training yapmak için kullanılacak.
4) Oluşturduğunuz datasetini veya datasetlerini kullanarak DataLoader (torch) oluşturun.
5) train.py den ise train için yazdığımız metodu dataloaderlerinizle seçtiğiniz öğrenme parametreleriyle çalıştırın.