import os
from datetime import datetime

import pytz
from django.test import TestCase
from django_rq import get_worker
from rest_framework.test import APIClient

from api.api_util import get_search_term_examples

# from api.directory_watcher import scan_photos
from api.models import AlbumAuto, User

# To-Do: Fix setting IMAGE_DIRS and try scanning something
samplephotos_dir = os.path.abspath("samplephotos")


# Create your tests here.
class AdminTestCase(TestCase):
    def setUp(self):
        User.objects.create_superuser(
            "test_admin", "test_admin@test.com", "test_password"
        )
        self.client = APIClient()
        auth_res = self.client.post(
            "/api/auth/token/obtain/",
            {"username": "test_admin", "password": "test_password"},
        )
        self.client.credentials(
            HTTP_AUTHORIZATION="Bearer " + auth_res.json()["access"]
        )

    def test_admin_exists(self):
        test_admin = User.objects.get(username="test_admin")
        self.assertTrue(test_admin.is_superuser)

    def test_admin_login(self):
        res = self.client.post(
            "/api/auth/token/obtain/",
            {"username": "test_admin", "password": "test_password"},
        )
        self.assertEqual(res.status_code, 200)
        self.assertTrue("access" in res.json().keys())

    def test_list_directories(self):
        res = self.client.get("/api/dirtree/")
        self.assertEqual(res.status_code, 200)

    def test_get_albums_date_list(self):
        res = self.client.get("/api/albums/date/photohash/list/")
        self.assertEqual(res.status_code, 200)


class UserTestCase(TestCase):
    def setUp(self):
        self.client_admin = APIClient()
        self.client_user = APIClient()

        User.objects.create_superuser(
            "test_admin", "test_admin@test.com", "test_password"
        )
        admin_auth_res = self.client_admin.post(
            "/api/auth/token/obtain/",
            {"username": "test_admin", "password": "test_password"},
        )
        self.client_admin.credentials(
            HTTP_AUTHORIZATION="Bearer " + admin_auth_res.json()["access"]
        )

        # signup disabled by default
        create_user_res = self.client_user.post(
            "/api/user/", {"username": "test_admin", "password": "test_password"}
        )
        self.assertEqual(create_user_res.status_code, 401)

        # enable signup as admin
        change_settings_res = self.client_admin.post(
            "/api/sitesettings/", {"allow_registration": True}
        )
        self.assertEqual(change_settings_res.status_code, 200)

        # normal user is gonna try and set his own scan directory (which isn't allowed)
        forced_scan_directory = "/root/l33t/"

        # try signing up as a normal user again
        create_user_res = self.client_user.post(
            "/api/user/",
            {
                "username": "test_user",
                "email": "test_user@test.com",
                "password": "test_password",
                "scan_directory": forced_scan_directory,
            },
        )

        self.assertEqual(create_user_res.status_code, 201)
        self.assertFalse("password" in create_user_res.json().keys())

        # make sure setting his own scan_directory didn't work
        self.assertTrue(
            create_user_res.json()["scan_directory"] != forced_scan_directory
        )

        test_user_pk = create_user_res.json()["id"]

        # login as test_user
        user_auth_res = self.client_user.post(
            "/api/auth/token/obtain/",
            {"username": "test_user", "password": "test_password"},
        )
        self.client_user.credentials(
            HTTP_AUTHORIZATION="Bearer " + user_auth_res.json()["access"]
        )

        # make sure the logged in user cannot update his own scan_directory path
        patch_res = self.client_user.patch(
            f"/api/user/{test_user_pk}/", {"scan_directory": forced_scan_directory}
        )

        self.assertTrue(patch_res.json()["scan_directory"] != forced_scan_directory)

        # make sure get /api/user/ doesn't return password
        res = self.client.get("/api/user/")
        self.assertEqual(res.status_code, 200)
        for r in res.json()["results"]:
            self.assertFalse("password" in r.keys(), "Get user returned password")

    def test_get_albums_date_list(self):
        res = self.client_user.get("/api/albums/date/photohash/list/")
        self.assertEqual(res.status_code, 200)


class GetSearchTermExamples(TestCase):
    def test_get_search_term_examples(self):
        admin = User.objects.create_superuser(
            "test_admin", "test_admin@test.com", "test_password"
        )
        array = get_search_term_examples(admin)
        self.assertEqual(len(array), 5)


class RegenerateTitlesTestCase(TestCase):
    def test_regenerate_titles(self):
        admin = User.objects.create_superuser(
            "test_admin", "test_admin@test.com", "test_password"
        )
        # create a album auto
        album_auto = AlbumAuto.objects.create(
            timestamp=datetime.strptime("2022-01-02", "%Y-%m-%d").replace(
                tzinfo=pytz.utc
            ),
            created_on=datetime.strptime("2022-01-02", "%Y-%m-%d").replace(
                tzinfo=pytz.utc
            ),
            owner=admin,
        )
        album_auto._generate_title()
        self.assertEqual(album_auto.title, "Sunday")


class SetupDirectoryTestCase(TestCase):
    userid = 0

    def setUp(self):
        self.client_admin = APIClient()

        user = User.objects.create_superuser(
            "test_admin", "test_admin@test.com", "test_password"
        )

        self.userid = user.id
        admin_auth_res = self.client_admin.post(
            "/api/auth/token/obtain/",
            {
                "username": "test_admin",
                "password": "test_password",
            },
        )
        self.client_admin.credentials(
            HTTP_AUTHORIZATION="Bearer " + admin_auth_res.json()["access"]
        )

    def test_setup_directory(self):
        patch_res = self.client_admin.patch(
            f"/api/manage/user/{self.userid}/", {"scan_directory": "/code"}
        )

        self.assertEqual(patch_res.status_code, 200)

    def test_setup_not_existing_directory(self):
        patch_res = self.client_admin.patch(
            f"/api/manage/user/{self.userid}/",
            {"scan_directory": "/code/not/existing"},
        )

        self.assertEqual(patch_res.status_code, 400)


class ScanPhotosTestCase(TestCase):
    def setUp(self):
        self.client_admin = APIClient()

        self.client_users = [APIClient() for _ in range(2)]

        User.objects.create_superuser(
            "test_admin", "test_admin@test.com", "test_password"
        )
        admin_auth_res = self.client_admin.post(
            "/api/auth/token/obtain/",
            {
                "username": "test_admin",
                "password": "test_password",
            },
        )
        self.client_admin.credentials(
            HTTP_AUTHORIZATION="Bearer " + admin_auth_res.json()["access"]
        )

        # enable signup as admin
        change_settings_res = self.client_admin.post(
            "/api/sitesettings/", {"allow_registration": True}
        )
        self.assertEqual(change_settings_res.json()["allow_registration"], "True")
        self.assertEqual(change_settings_res.status_code, 200)

        logged_in_clients = []

        # sign up 6 test users
        user_ids = []

        for idx, client in enumerate(self.client_users):
            create_user_res = client.post(
                "/api/user/",
                {
                    "email": f"test_user_{idx}@test.com",
                    "username": f"test_user_{idx}",
                    "password": "test_password",
                },
            )


            self.assertEqual(create_user_res.status_code, 201)
            user_ids.append(create_user_res.json()["id"])

            login_user_res = client.post(
                "/api/auth/token/obtain/",
                {"username": f"test_user_{idx}", "password": "test_password"},
            )

            self.assertEqual(login_user_res.status_code, 200)

            client.credentials(
                HTTP_AUTHORIZATION="Bearer " + login_user_res.json()["access"]
            )
            logged_in_clients.append(client)
        self.client_users = logged_in_clients

        # set scan directories for each user as admin
        for idx, (user_id, client) in enumerate(zip(user_ids, self.client_users)):

            user_scan_directory = os.path.join(samplephotos_dir, f"test{idx}")
            self.assertNotEqual(user_scan_directory, "")
            patch_res = self.client_admin.patch(
                f"/api/manage/user/{user_id}/",
                {"scan_directory": user_scan_directory},
            )

            self.assertEqual(patch_res.json(), {})
            self.assertEqual(patch_res.status_code, 200)
            self.assertEqual(patch_res.json()["scan_directory"], user_scan_directory)

        # make sure users are logged in
        for client in self.client_users:
            res = client.get("/api/photos/")
            self.assertEqual(res.status_code, 200)

        # scan photos
        scan_photos_res = self.client_users[0].get("/api/scanphotos/")
        self.assertEqual(scan_photos_res.status_code, 200)
        get_worker().work(burst=True)

        # make sure photos are imported
        get_photos_res = self.client_users[0].get("/api/photos/")
        self.assertEqual(get_photos_res.status_code, 200)
        self.assertTrue(len(get_photos_res.json()["results"]) > 0)

        # try scanning again and make sure there are no duplicate imports
        num_photos = len(get_photos_res.json()["results"])
        scan_photos_res = self.client_users[0].get("/api/scanphotos/")
        self.assertEqual(scan_photos_res.status_code, 200)
        get_worker().work(burst=True)
        get_photos_res = self.client_users[0].get("/api/photos/")
        self.assertEqual(get_photos_res.status_code, 200)
        self.assertEqual(len(get_photos_res.json()["results"]), num_photos)

    def test_auto_albums(self):
        """make sure user can make auto albums, list and retrieve them"""
        # make auto albums
        auto_album_gen_res = self.client_users[0].get("/api/autoalbumgen/")
        self.assertEqual(auto_album_gen_res.status_code, 200)
        get_worker().work(burst=True)

        # make sure auto albums are there
        auto_album_list_res = self.client_users[0].get("/api/albums/auto/list/")
        self.assertEqual(auto_album_list_res.status_code, 200)

        # make sure user can retrieve each auto album
        for album in auto_album_list_res.json()["results"]:
            auto_album_retrieve_res = self.client_users[0].get(
                "/api/albums/auto/%d/" % album["id"]
            )
            self.assertEqual(auto_album_retrieve_res.status_code, 200)
            self.assertTrue(len(auto_album_retrieve_res.json()["photos"]) > 0)

        # try making auto albums again and make sure there are no duplicates
        num_auto_albums = len(auto_album_list_res.json()["results"])

        auto_album_gen_res = self.client_users[0].get("/api/autoalbumgen/")
        self.assertEqual(auto_album_gen_res.status_code, 200)
        get_worker().work(burst=True)

        auto_album_list_res = self.client_users[0].get("/api/albums/auto/list/")
        self.assertEqual(len(auto_album_list_res.json()["results"]), num_auto_albums)

    def test_place_albums(self):
        """make sure user can list and retrieve place albums"""
        place_album_list_res = self.client_users[0].get("/api/albums/place/list/")
        self.assertEqual(place_album_list_res.status_code, 200)

        for album in place_album_list_res.json()["results"]:
            place_album_retrieve_res = self.client_users[0].get(
                "/api/albums/place/%d/" % album["id"]
            )
            self.assertEqual(place_album_retrieve_res.status_code, 200)

    def test_thing_albums(self):
        """make sure user can list and retrieve thing albums"""
        thing_album_list_res = self.client_users[0].get("/api/albums/thing/list/")
        self.assertEqual(thing_album_list_res.status_code, 200)

        for album in thing_album_list_res.json()["results"]:
            thing_album_retrieve_res = self.client_users[0].get(
                "/api/albums/thing/%d/" % album["id"]
            )
            self.assertEqual(thing_album_retrieve_res.status_code, 200)
