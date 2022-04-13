import itertools

import face_recognition
import matplotlib.pyplot as plt
import numpy as np
from seaborn import color_palette
from sklearn import mixture, preprocessing
from sklearn.cluster import (
    DBSCAN,
    AgglomerativeClustering,
    MeanShift,
    estimate_bandwidth,
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from api.models import Face, Person


def get_or_create_person(name):
    qs = Person.objects.filter(name=name)
    if qs.count() > 0:
        return qs[0]
    new_person = Person()
    new_person.name = name
    new_person.save()
    return new_person


def get_face_encoding(face):
    return np.frombuffer(bytes.fromhex(face.encoding))


def nuke_people():
    for person in Person.objects.filter(name__startswith="Person"):
        person.delete()


faces = list(Face.objects.all())
face_encodings = np.array([np.frombuffer(bytes.fromhex(f.encoding)) for f in faces])

num_groups = []
for _ in tqdm(range(50)):
    groups = []
    np.random.shuffle(faces)
    for face in faces:
        if not groups:
            groups.append([face])
        else:
            group_this_face_belongs_to = None
            encoding_face_curr = get_face_encoding(face)

            for group_idx, group in enumerate(groups):
                face_group_repr = group[0]
                encoding_face_group_repr = get_face_encoding(face_group_repr)
                if face_recognition.compare_faces(
                    [encoding_face_group_repr], encoding_face_curr, tolerance=0.65
                )[0]:
                    group_this_face_belongs_to = group_idx

            if group_this_face_belongs_to:
                groups[group_this_face_belongs_to].append(face)
            else:
                groups.append([face])
    num_groups.append(len(groups))

num_people = int(np.mean(num_groups))


nuke_people()
faces = list(Face.objects.all())
face_encodings = np.array([np.frombuffer(bytes.fromhex(f.encoding)) for f in faces])
X = StandardScaler().fit_transform(face_encodings)

bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)


# p = Photo.objects.first()
# image_path = p.image_path
# captions = {}
# with open(image_path, "rb") as image_file:
#     encoded_string = base64.b64encode(image_file.read())
# encoded_string = str(encoded_string)[2:-1]
# resp_captions = requests.post('http://localhost:5001/longcaptions/',data=encoded_string)


# faces = Face.objects.all()
# face_encodings = [np.frombuffer(bytes.fromhex(f.encoding)) for f in faces]
# person_ids = [f.person.id for f in faces]
# palette = color_palette('Paired',max(person_ids)+1).as_hex()
# colors = [palette[i] for i in person_ids]

# face_embedded = TSNE(n_components=2,n_iter=100000,verbose=1,perplexity=50).fit_transform(face_encodings)
# plt.scatter(face_embedded[:,0],face_embedded[:,1],c=colors)
# plt.show()


# start = datetime.now()
# qs = AlbumDate.objects.all().order_by('date').prefetch_related(
#     Prefetch('photos', queryset=Photo.objects.all().only('image_hash','exif_timestamp','favorited','hidden')))
# qs_res = list(qs)
# print('db query took %.2f seconds'%(datetime.now()-start).total_seconds())

# start = datetime.now()
# res = AlbumDateListWithPhotoHashSerializerSerpy(qs_res,many=True).data
# print('serpy serializing took %.2f seconds'%(datetime.now()-start).total_seconds())

# start = datetime.now()
# res = AlbumDateListWithPhotoHashSerializer(qs_res,many=True).data
# print('drf serializing took %.2f seconds'%(datetime.now()-start).total_seconds())


# SELECT ("api_albumdate_photos"."albumdate_id") AS "_prefetch_related_val_albumdate_id",
#        "api_photo"."image_hash",
#        "api_photo"."exif_timestamp",
#        "api_photo"."favorited",
#        "api_photo"."hidden"
#   FROM "api_photo"
#  INNER JOIN "api_albumdate_photos"
#     ON ("api_photo"."image_hash" = "api_albumdate_photos"."photo_id");
