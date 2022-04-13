import serpy
from django.core.cache import cache
from rest_framework import serializers

from api.models import AlbumUser, Photo
from api.serializers.PhotosGroupedByDate import get_photos_ordered_by_date
from api.serializers.serializers import PhotoHashListSerializer
from api.serializers.serializers_serpy import GroupedPhotosSerializer
from api.serializers.user import SimpleUserSerializer, SimpleUserSerializerSerpy
from api.util import logger


class AlbumUserSerializerSerpy(serpy.Serializer):
    id = serpy.StrField()
    title = serpy.StrField()
    owner = SimpleUserSerializerSerpy()
    shared_to = SimpleUserSerializerSerpy(many=True, call=True, attr="shared_to.all")
    date = serpy.MethodField("get_date")
    location = serpy.MethodField("get_location")
    grouped_photos = serpy.MethodField("get_photos")

    def get_photos(self, obj):
        grouped_photos = get_photos_ordered_by_date(
            obj.photos.all().order_by("-exif_timestamp")
        )
        return GroupedPhotosSerializer(grouped_photos, many=True).data

    def get_location(self, obj):
        return next(
            (
                photo.search_location
                for photo in obj.photos.all()
                if photo and photo.search_location
            ),
            "",
        )

    def get_date(self, obj):
        return next(
            (
                photo.exif_timestamp
                for photo in obj.photos.all()
                if photo and photo.exif_timestamp
            ),
            "",
        )


class AlbumUserEditSerializer(serializers.ModelSerializer):
    photos = serializers.PrimaryKeyRelatedField(
        many=True, read_only=False, queryset=Photo.objects.all()
    )
    removedPhotos = serializers.ListField(
        child=serializers.CharField(max_length=100, default=""),
        write_only=True,
        required=False,
    )

    class Meta:
        model = AlbumUser
        fields = ("id", "title", "photos", "created_on", "favorited", "removedPhotos")

    def validate_photos(self, value):
        return [v.image_hash for v in value]

    def create(self, validated_data):
        title = validated_data["title"]
        image_hashes = validated_data["photos"]
        request = self.context.get("request")
        user = request.user if request and hasattr(request, "user") else None
        # check if an album exists with the given title and call the update method if it does
        instance, created = AlbumUser.objects.get_or_create(title=title, owner=user)
        if not created:
            return self.update(instance, validated_data)

        photos = Photo.objects.in_bulk(image_hashes)
        for pk, obj in photos.items():
            instance.photos.add(obj)
        instance.save()
        cache.clear()
        logger.info(f"Created user album {instance.id} with {len(photos)} photos")
        return instance

    def update(self, instance, validated_data):

        if "title" in validated_data.keys():
            title = validated_data["title"]
            instance.title = title
            logger.info(f"Renamed user album to {title}")

        if "removedPhotos" in validated_data.keys():
            image_hashes = validated_data["removedPhotos"]
            photos_already_in_album = instance.photos.all()
            cnt = 0
            for obj in photos_already_in_album:
                if obj.image_hash in image_hashes:
                    cnt += 1
                    instance.photos.remove(obj)

            logger.info(f"Removed {cnt} photos to user album {instance.id}")

        if "photos" in validated_data.keys():
            image_hashes = validated_data["photos"]
            photos = Photo.objects.in_bulk(image_hashes)
            photos_already_in_album = instance.photos.all()
            cnt = 0
            for pk, obj in photos.items():
                if obj not in photos_already_in_album:
                    cnt += 1
                    instance.photos.add(obj)

            logger.info(f"Added {cnt} photos to user album {instance.id}")

        cache.clear()
        instance.save()
        return instance


class AlbumUserListSerializer(serializers.ModelSerializer):
    cover_photos = PhotoHashListSerializer(many=True, read_only=True)
    photo_count = serializers.SerializerMethodField()
    shared_to = SimpleUserSerializer(many=True, read_only=True)
    owner = SimpleUserSerializer(many=False, read_only=True)

    class Meta:
        model = AlbumUser
        fields = (
            "id",
            "cover_photos",
            "created_on",
            "favorited",
            "title",
            "shared_to",
            "owner",
            "photo_count",
        )

    def get_photo_count(self, obj):
        try:
            return obj.photo_count
        except Exception:  # for when calling AlbumUserListSerializer(obj).data directly
            return obj.photos.count()
