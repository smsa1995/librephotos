from datetime import datetime, timedelta

import numpy as np
import pytz
from django.core.cache import cache
from django.db.models import Q
from django_rq import job

from api.models import (
    AlbumAuto,
    AlbumDate,
    AlbumPlace,
    AlbumThing,
    AlbumUser,
    Face,
    LongRunningJob,
    Photo,
)
from api.util import logger


@job
def regenerate_event_titles(user, job_id):
    if LongRunningJob.objects.filter(job_id=job_id).exists():
        lrj = LongRunningJob.objects.get(job_id=job_id)
        lrj.started_at = datetime.now().replace(tzinfo=pytz.utc)
    else:
        lrj = LongRunningJob.objects.create(
            started_by=user,
            job_id=job_id,
            queued_at=datetime.now().replace(tzinfo=pytz.utc),
            started_at=datetime.now().replace(tzinfo=pytz.utc),
            job_type=LongRunningJob.JOB_GENERATE_AUTO_ALBUM_TITLES,
        )
    lrj.save()
    try:
        aus = AlbumAuto.objects.filter(owner=user).prefetch_related("photos")
        target_count = len(aus)
        for idx, au in enumerate(aus):
            logger.info(f"job {job_id}: {idx}")
            au._generate_title()
            au.save()

            lrj.result = {"progress": {"current": idx + 1, "target": target_count}}
            lrj.save()

        lrj.finished = True
        lrj.finished_at = datetime.now().replace(tzinfo=pytz.utc)
        lrj.save()
        logger.info(f"job {job_id}: updated lrj entry to db")

    except Exception:
        logger.exception("An error occured")
        lrj.failed = True
        lrj.finished = True
        lrj.finished_at = datetime.now().replace(tzinfo=pytz.utc)
        lrj.save()

    return 1


@job
def generate_event_albums(user, job_id):
    if LongRunningJob.objects.filter(job_id=job_id).exists():
        lrj = LongRunningJob.objects.get(job_id=job_id)
        lrj.started_at = datetime.now().replace(tzinfo=pytz.utc)
        lrj.save()
    else:
        lrj = LongRunningJob.objects.create(
            started_by=user,
            job_id=job_id,
            queued_at=datetime.now().replace(tzinfo=pytz.utc),
            started_at=datetime.now().replace(tzinfo=pytz.utc),
            job_type=LongRunningJob.JOB_GENERATE_AUTO_ALBUMS,
        )
        lrj.save()

    try:
        photos = (
            Photo.objects.filter(Q(owner=user))
            .exclude(Q(exif_timestamp=None))
            .only("exif_timestamp")
        )

        photos_with_timestamp = [
            (photo.exif_timestamp, photo) for photo in photos if photo.exif_timestamp
        ]

        def group(photos_with_timestamp, dt=timedelta(hours=6)):
            photos_with_timestamp = sorted(photos_with_timestamp, key=lambda x: x[0])
            groups = []
            for idx, photo in enumerate(photos_with_timestamp):
                if not groups:
                    groups.append([])
                elif photo[0] - groups[-1][-1].exif_timestamp >= dt:
                    groups.append([])
                groups[-1].append(photo[1])
                logger.info(f"job {job_id}: {idx}")
            return groups

        groups = group(photos_with_timestamp, dt=timedelta(days=1, hours=12))
        logger.info("job {}: made groups".format(job_id))

        album_locations = []

        target_count = len(groups)

        date_format = "%Y:%m:%d %H:%M:%S"
        for idx, group in enumerate(groups):
            key = group[0].exif_timestamp - timedelta(hours=11, minutes=59)
            lastKey = group[-1].exif_timestamp + timedelta(hours=11, minutes=59)
            logger.info(str(key.date) + " - " + str(lastKey.date))
            logger.info(
                "job {}: processing auto album with date: ".format(job_id)
                + key.strftime(date_format)
                + " to "
                + lastKey.strftime(date_format)
            )
            items = group
            if len(group) >= 2:
                qs = AlbumAuto.objects.filter(owner=user).filter(
                    timestamp__range=(key, lastKey)
                )
                if qs.count() == 0:
                    album = AlbumAuto(
                        created_on=datetime.utcnow().replace(tzinfo=pytz.utc),
                        owner=user,
                    )
                    album.timestamp = key
                    album.save()

                    locs = []
                    for item in items:
                        album.photos.add(item)
                        item.save()
                        if item.exif_gps_lat and item.exif_gps_lon:
                            locs.append([item.exif_gps_lat, item.exif_gps_lon])
                    if len(locs) > 0:
                        album_location = np.mean(np.array(locs), 0)
                        album_locations.append(album_location)
                        album.gps_lat = album_location[0]
                        album.gps_lon = album_location[1]
                    else:
                        album_locations.append([])
                    album._generate_title()
                    album.save()
                    logger.info(
                        "job {}: generated auto album {}".format(job_id, album.id)
                    )
                if qs.count() == 1:
                    album = qs.first()
                    for item in items:
                        if item in album.photos.all():
                            continue
                        album.photos.add(item)
                        item.save()
                    album._generate_title()
                    album.save()
                    logger.info(
                        "job {}: updated auto album {}".format(job_id, album.id)
                    )
                if qs.count() > 1:
                    # To-Do: Merge both auto albums
                    logger.info(
                        "job {}: found multiple auto albums for date {}".format(
                            job_id, key.strftime(date_format)
                        )
                    )

            lrj.result = {"progress": {"current": idx + 1, "target": target_count}}
            lrj.save()

        lrj.finished = True
        lrj.finished_at = datetime.now().replace(tzinfo=pytz.utc)
        lrj.save()

    except Exception:
        logger.exception("An error occured")
        lrj.failed = True
        lrj.finished = True
        lrj.finished_at = datetime.now().replace(tzinfo=pytz.utc)
        lrj.save()

    return 1


@job
def delete_missing_photos(user, job_id):
    if LongRunningJob.objects.filter(job_id=job_id).exists():
        lrj = LongRunningJob.objects.get(job_id=job_id)
        lrj.started_at = datetime.now().replace(tzinfo=pytz.utc)
    else:
        lrj = LongRunningJob.objects.create(
            started_by=user,
            job_id=job_id,
            queued_at=datetime.now().replace(tzinfo=pytz.utc),
            started_at=datetime.now().replace(tzinfo=pytz.utc),
            job_type=LongRunningJob.JOB_DELETE_MISSING_PHOTOS,
        )
    lrj.save()
    try:
        missing_photos = Photo.objects.filter(Q(owner=user) & Q(image_paths=[]))
        for missing_photo in missing_photos:
            album_dates = AlbumDate.objects.filter(photos=missing_photo)
            for album_date in album_dates:
                album_date.photos.remove(missing_photo)
            album_things = AlbumThing.objects.filter(photos=missing_photo)
            for album_thing in album_things:
                album_thing.photos.remove(missing_photo)
            album_places = AlbumPlace.objects.filter(photos=missing_photo)
            for album_place in album_places:
                album_place.photos.remove(missing_photo)
            album_users = AlbumUser.objects.filter(photos=missing_photo)
            for album_user in album_users:
                album_user.photos.remove(missing_photo)
            faces = Face.objects.filter(photo=missing_photo)
            faces.delete()
        missing_photos.delete()
        cache.clear()

        lrj.finished = True
        lrj.finished_at = datetime.now().replace(tzinfo=pytz.utc)
        lrj.save()
    except Exception:
        logger.exception("An error occured")
        lrj.failed = True
        lrj.finished = True
        lrj.finished_at = datetime.now().replace(tzinfo=pytz.utc)
        lrj.save()
    return 1
