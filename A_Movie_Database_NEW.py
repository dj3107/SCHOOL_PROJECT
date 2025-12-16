import sys
import time
import webbrowser

import numpy as np
import requests
import sounddevice as sd
import whisper

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QFont, QFontInfo, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QListWidget,
    QLabel,
    QTextBrowser,
    QMessageBox,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QSplitter,
    QDialog,
    QScrollArea,
    QGridLayout,
    QListWidgetItem,
)

# =================================================
# STYLING & THEME
# =================================================

STYLESHEET = """
/* Apple Dark Mode Theme */

QMainWindow {
    background-color: #000000;
    color: #f5f5f7;
    font-family: 'SF Pro Display', 'Helvetica Neue', sans-serif;
}

QDialog {
    background-color: #000000;
    color: #f5f5f7;
    font-family: 'SF Pro Display', 'Helvetica Neue', sans-serif;
}

QScrollArea {
    background-color: #000000;
    border: none;
}

QTabWidget::pane {
    border: 1px solid #424245;
}

QTabBar::tab {
    background-color: #1d1d1f;
    color: #f5f5f7;
    padding: 10px 20px;
    border: 1px solid #424245;
    border-bottom: none;
    font-size: 14px;
    font-weight: 500;
    font-family: 'SF Pro Text', sans-serif;
}

QTabBar::tab:selected {
    background-color: #2a2a2d;
    border-bottom: 2px solid #0a84ff;
    color: #0a84ff;
}

QLineEdit, QTextBrowser {
    background-color: #1d1d1f;
    color: #f5f5f7;
    border: 1px solid #424245;
    border-radius: 8px;
    padding: 12px;
    font-size: 14px;
    font-family: 'SF Pro Text', sans-serif;
}

QLineEdit:focus {
    border: 2px solid #0a84ff;
}

QPushButton {
    background-color: #0a84ff;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 12px 20px;
    font-weight: 600;
    font-size: 14px;
    font-family: 'SF Pro Text', sans-serif;
}

QPushButton:hover {
    background-color: #0060df;
}

QPushButton:pressed {
    background-color: #004ab3;
}

QListWidget {
    background-color: #1d1d1f;
    color: #f5f5f7;
    border: 1px solid #424245;
    border-radius: 8px;
    font-size: 18px;
    font-family: 'SF Pro Text', sans-serif;
}

QListWidget::item {
    padding: 10px 8px;
}

QListWidget::item:selected {
    background-color: #0a84ff;
    color: #ffffff;
}

QLabel {
    color: #f5f5f7;
    font-size: 14px;
    padding: 8px;
    font-family: 'SF Pro Text', sans-serif;
}

QTreeWidget {
    background-color: #1d1d1f;
    color: #f5f5f7;
    border: 1px solid #424245;
    border-radius: 8px;
    font-size: 14px;
    font-family: 'SF Pro Text', sans-serif;
}

QTreeWidget::item {
    padding: 8px;
}

QTreeWidget::item:selected {
    background-color: #0a84ff;
    color: #ffffff;
}

QSplitter::handle {
    background-color: #424245;
    width: 4px;
}
"""

# =================================================
# API KEYS / CONSTANTS
# =================================================

OMDB_KEY = "cc0fba9e"
TMDB_KEY = "b3a56cb04a5f02d1229c0fc11b3303c2"

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w300"
TMDB_IMAGE_LARGE = "https://image.tmdb.org/t/p/w500"


OMDB_BASE_URL = "http://www.omdbapi.com/"
TMDB_BASE_URL = "https://api.themoviedb.org/3"


# =================================================
# OMDb HELPERS (IMDb ratings only)
# =================================================

def search_omdb(title, type_, api_key=OMDB_KEY):
    if not title:
        return []
    params = {"apikey": api_key, "s": title, "type": type_}
    try:
        r = requests.get(OMDB_BASE_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data and data.get("Response") == "True":
            return data.get("Search", []) or []
        return []
    except Exception:
        return []


def get_omdb_by_imdb_id(imdb_id, plot="full"):
    """
    Fetch details (including IMDb rating) from OMDb by IMDb ID.
    """
    if not imdb_id:
        return None
    params = {
        "apikey": OMDB_KEY,
        "i": imdb_id,
        "plot": plot,
    }
    try:
        r = requests.get(OMDB_BASE_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data and data.get("Response") == "True":
            return data
        return None
    except Exception:
        return None


def get_omdb_details(title, type_):
    """
    Legacy helper: look up by title + type.
    Kept for compatibility but movie/TV pages prefer get_omdb_by_imdb_id.
    """
    if not title:
        return None
    params = {
        "apikey": OMDB_KEY,
        "t": title,
        "type": type_,
        "plot": "full",
    }
    try:
        r = requests.get(OMDB_BASE_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data and data.get("Response") == "True":
            return data
        return None
    except Exception:
        return None


def get_omdb_episode_reviews(imdb_id):
    """
    Fetch an episode's OMDb record by IMDb ID.
    (Used for episode IMDb ratings / plots.)
    """
    return get_omdb_by_imdb_id(imdb_id, plot="full")


# =================================================
# TMDb HELPERS (NO TMDb SCORES SHOWN TO USER)
# =================================================

def _tmdb_get(path, params=None):
    if params is None:
        params = {}
    params = dict(params)
    params["api_key"] = TMDB_KEY
    try:
        url = f"{TMDB_BASE_URL}/{path.lstrip('/')}"
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json() or {}
    except Exception:
        return {}


def search_tmdb_movie(query, page=1):
    if not query:
        return []
    data = _tmdb_get("search/movie", {"query": query, "page": page, "sort_by": "popularity.desc"})
    results = data.get("results", []) or []
    # Internally sort by popularity, but we never display that numeric score.
    return sorted(results, key=lambda x: x.get("popularity", 0), reverse=True)


def search_tmdb_tv(query, page=1):
    if not query:
        return []
    data = _tmdb_get("search/tv", {"query": query, "page": page, "sort_by": "popularity.desc"})
    results = data.get("results", []) or []
    return sorted(results, key=lambda x: x.get("popularity", 0), reverse=True)


def search_tmdb_person(query, page=1):
    if not query:
        return []
    data = _tmdb_get("search/person", {"query": query, "page": page})
    results = data.get("results", []) or []
    return sorted(results, key=lambda x: x.get("popularity", 0), reverse=True)


def search_tmdb_keyword(query, page=1):
    if not query:
        return []
    data = _tmdb_get("search/keyword", {"query": query, "page": page})
    return data.get("results", []) or []


def discover_by_keyword(keyword_id, page=1):
    if not keyword_id:
        return []
    data = _tmdb_get("discover/movie", {"with_keywords": keyword_id, "page": page})
    return data.get("results", []) or []


def get_tmdb_movie_details(tmdb_id):
    if not tmdb_id:
        return {}
    return _tmdb_get(f"movie/{tmdb_id}")


def get_tmdb_tv_details(tmdb_id):
    if not tmdb_id:
        return {}
    return _tmdb_get(f"tv/{tmdb_id}")


def get_tmdb_tv_season_episodes(tmdb_id, season_number):
    if not tmdb_id:
        return []
    data = _tmdb_get(f"tv/{tmdb_id}/season/{season_number}")
    return data.get("episodes", []) or []


def get_tmdb_movie_videos(tmdb_id):
    if not tmdb_id:
        return []
    data = _tmdb_get(f"movie/{tmdb_id}/videos")
    return data.get("results", []) or []


def get_tmdb_tv_videos(tmdb_id):
    if not tmdb_id:
        return []
    data = _tmdb_get(f"tv/{tmdb_id}/videos")
    return data.get("results", []) or []


def get_tmdb_movie_credits(tmdb_id):
    if not tmdb_id:
        return {"cast": [], "crew": []}
    data = _tmdb_get(f"movie/{tmdb_id}/credits")
    return data or {"cast": [], "crew": []}


def get_tmdb_tv_credits(tmdb_id):
    if not tmdb_id:
        return {"cast": [], "crew": []}
    data = _tmdb_get(f"tv/{tmdb_id}/credits")
    return data or {"cast": [], "crew": []}


def get_tmdb_person_details(person_id):
    if not person_id:
        return None
    data = _tmdb_get(f"person/{person_id}")
    return data or None


def get_tmdb_person_filmography(person_id):
    """
    Returns dict with keys: cast, directing, writing, producing.
    """
    if not person_id:
        return {"cast": [], "directing": [], "writing": [], "producing": []}

    data = _tmdb_get(f"person/{person_id}/combined_credits")

    cast_credits = []
    directing = []
    writing = []
    producing = []

    for credit in data.get("cast", []) or []:
        cast_credits.append({
            "id": credit.get("id"),
            "title": credit.get("title") or credit.get("name", "Unknown"),
            "character": credit.get("character", ""),
            "type": "movie" if credit.get("media_type") == "movie" else "tv",
            "poster_path": credit.get("poster_path"),
            "release_date": credit.get("release_date") or credit.get("first_air_date", ""),
        })

    for credit in data.get("crew", []) or []:
        job = (credit.get("job") or "").lower()
        info = {
            "id": credit.get("id"),
            "title": credit.get("title") or credit.get("name", "Unknown"),
            "job": credit.get("job", "N/A"),
            "type": "movie" if credit.get("media_type") == "movie" else "tv",
            "poster_path": credit.get("poster_path"),
            "release_date": credit.get("release_date") or credit.get("first_air_date", ""),
        }
        if "director" in job:
            directing.append(info)
        elif "writer" in job or "screenplay" in job:
            writing.append(info)
        elif "producer" in job:
            producing.append(info)

    for lst in (cast_credits, directing, writing, producing):
        lst.sort(key=lambda x: x.get("release_date", "0000"), reverse=True)

    return {
        "cast": cast_credits,
        "directing": directing[:30],
        "writing": writing[:30],
        "producing": producing[:30],
    }


def find_common_filmography_with_roles(people_ids):
    """
    Find films where ALL specified people worked together, with their roles.
    Returns list of [tmdb_id, {person_index: [roles,...]}]
    """
    if not people_ids or len(people_ids) < 2:
        return []

    try:
        all_filmographies = []
        film_roles = {}

        for idx, person_id in enumerate(people_ids):
            person_films = {}
            filmography = get_tmdb_person_filmography(person_id)

            for c in filmography.get("cast", []):
                fid = c["id"]
                if fid not in person_films:
                    person_films[fid] = []
                person_films[fid].append("Actor")

            for c in filmography.get("directing", []):
                fid = c["id"]
                if fid not in person_films:
                    person_films[fid] = []
                person_films[fid].append("Director")

            for c in filmography.get("writing", []):
                fid = c["id"]
                if fid not in person_films:
                    person_films[fid] = []
                person_films[fid].append("Writer")

            for c in filmography.get("producing", []):
                fid = c["id"]
                if fid not in person_films:
                    person_films[fid] = []
                person_films[fid].append("Producer")

            all_filmographies.append(person_films)

        # Build film_roles: film_id -> idx -> roles
        for idx, person_films in enumerate(all_filmographies):
            for film_id_key, roles in person_films.items():
                if film_id_key not in film_roles:
                    film_roles[film_id_key] = {}
                film_roles[film_id_key][idx] = roles

        common_films = []
        for film_id, person_roles in film_roles.items():
            if len(person_roles) == len(people_ids):
                common_films.append([film_id, person_roles])

        return common_films
    except Exception:
        return []


def get_tmdb_recommendations(media_type="movie", limit=20):
    try:
        data = _tmdb_get(f"trending/{media_type}/week")
        return (data.get("results", []) or [])[:limit]
    except Exception:
        return []


def get_tmdb_external_ids(tmdb_id, media_type="movie"):
    if not tmdb_id:
        return {}
    data = _tmdb_get(f"{media_type}/{tmdb_id}/external_ids")
    return data or {}


def get_tmdb_episode_external_ids(tv_id, season_number, episode_number):
    """
    Get external IDs (including IMDb ID) for a specific TV episode.
    """
    if not tv_id:
        return {}
    data = _tmdb_get(f"tv/{tv_id}/season/{season_number}/episode/{episode_number}/external_ids")
    return data or {}


# =================================================
# WHISPER THREAD
# =================================================


# =================================================
# BOX OFFICE FORMATTING
# =================================================

def format_currency(value):
    """Format a value as USD currency. If value is 0 or None, return 'N/A'."""
    if not value or value == 0:
        return "N/A"
    try:
        return f"${value:,.0f}"
    except:
        return "N/A"

class WhisperRecordThread(QThread):
    final = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, model_name="large-v3-turbo"):
        super().__init__()
        self.model_name = model_name
        self.running = False
        try:
            self.model = whisper.load_model(self.model_name)
        except Exception as e:
            self.model = None
            self.error.emit(f"Could not load Whisper model: {e}")
        self.recorded_audio = []

    def run(self):
        if self.model is None:
            return

        samplerate = 16000
        channels = 1
        self.running = True
        self.recorded_audio = []

        def callback(indata, frames, time_info, status):
            if self.running:
                self.recorded_audio.append(indata.copy())

        try:
            with sd.InputStream(
                callback=callback,
                channels=channels,
                samplerate=samplerate,
                dtype="int16"
            ):
                while self.running:
                    time.sleep(0.1)
        except Exception as e:
            self.error.emit(str(e))
            return

        if len(self.recorded_audio) == 0:
            self.error.emit("No audio recorded.")
            return

        # same processing as ihyperp.py
        audio_np = np.concatenate(
            [a[:, 0] for a in self.recorded_audio]
        ).astype(np.float32) / 32768.0

        if np.max(np.abs(audio_np)) < 0.01:
            self.error.emit("Too quiet. No speech detected.")
            return

        try:
            result = self.model.transcribe(audio_np, fp16=False)
            text = result.get("text", "").strip()
            if not text:
                self.error.emit("Could not recognize speech.")
                return
            self.final.emit(text)
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self.running = False



# =================================================
# VOICE SEARCH WINDOW
# =================================================

class VoiceRecordWindow(QDialog):
    search_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🎤 Voice Search")
        self.resize(400, 200)
        self.setStyleSheet(STYLESHEET)

        layout = QVBoxLayout(self)

        self.status_label = QLabel("Click 'Start Recording' to begin...")
        self.status_label.setFont(QFont("Arial", 12))
        layout.addWidget(self.status_label)

        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("▶️ Start Recording")
        self.start_btn.clicked.connect(self.start_recording)
        button_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("⏹️ Stop Recording")
        self.stop_btn.clicked.connect(self.stop_recording)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)

        layout.addLayout(button_layout)

        self.result_browser = QTextBrowser()
        layout.addWidget(self.result_browser)

        self.whisper_thread = None

    def start_recording(self):
        self.status_label.setText("🔴 Recording... Click Stop when done")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.whisper_thread = WhisperRecordThread()
        self.whisper_thread.final.connect(self.on_transcription_done)
        self.whisper_thread.error.connect(self.on_transcription_error)
        self.whisper_thread.start()

    def stop_recording(self):
        if self.whisper_thread:
            self.whisper_thread.stop()
            self.status_label.setText("⏳ Processing...")
            self.stop_btn.setEnabled(False)

    def on_transcription_done(self, text):
        self.status_label.setText("✅ Recognition Complete")
        self.result_browser.setText(f"\n\nYou said:\n{text}")
        self.search_requested.emit(text)
        self.start_btn.setEnabled(True)

    def on_transcription_error(self, error):
        self.status_label.setText(f"❌ Error: {error}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)


# =================================================
# SEARCH WINDOW
# =================================================

class SearchWindow(QDialog):
    item_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🔍 Search")
        self.resize(1200, 700)
        self.setStyleSheet(STYLESHEET)

        layout = QVBoxLayout(self)

        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search movies, TV shows, people...")
        search_layout.addWidget(self.search_input)

        search_btn = QPushButton("🔎 Click to Search")
        search_btn.clicked.connect(self.on_search_button_clicked)
        search_layout.addWidget(search_btn)

        layout.addLayout(search_layout)

        self.tabs = QTabWidget()

        self.movies_list = QListWidget()
        self.movies_list.itemClicked.connect(self.on_movie_selected)
        self.tabs.addTab(self.movies_list, "🎬 Movies")

        self.tv_list = QListWidget()
        self.tv_list.itemClicked.connect(self.on_tv_selected)
        self.tabs.addTab(self.tv_list, "📺 TV Shows")

        self.people_list = QListWidget()
        self.people_list.itemClicked.connect(self.on_people_selected)
        self.tabs.addTab(self.people_list, "👤 People")

        self.collaborations_list = QListWidget()
        self.collaborations_list.itemClicked.connect(self.on_collaboration_selected)
        self.tabs.addTab(self.collaborations_list, "🤝 Collaborations")

        layout.addWidget(self.tabs)

        self.people_ids = []

    def on_search_button_clicked(self):
        query = self.search_input.text().strip()
        if len(query) < 2:
            QMessageBox.warning(self, "Invalid Search", "Please enter at least 2 characters")
            return
        self.perform_search(query)

    def perform_search(self, query):
        seen_ids = set()

        # =========================
        # Movies (sorted by popularity)
        # =========================
        movies = search_tmdb_movie(query)
        movies = sorted(movies, key=lambda m: m.get("popularity", 0) or 0, reverse=True)
        self.movies_list.clear()
        for movie in movies[:20]:
            movie_id = movie.get("id")
            if movie_id in seen_ids:
                continue
            seen_ids.add(movie_id)
            title = movie.get("title", "N/A")
            year = (movie.get("release_date") or "")[:4] if movie.get("release_date") else "N/A"
            # Do NOT display popularity value, only use it for sorting.
            text = f"{title} ({year})"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, movie_id)
            self.movies_list.addItem(item)

        # =========================
        # TV Shows (sorted by popularity)
        # =========================
        tv_shows = search_tmdb_tv(query)
        tv_shows = sorted(tv_shows, key=lambda s: s.get("popularity", 0) or 0, reverse=True)
        self.tv_list.clear()
        for show in tv_shows[:20]:
            show_id = show.get("id")
            if show_id in seen_ids:
                continue
            seen_ids.add(show_id)
            title = show.get("name", "N/A")
            year = (show.get("first_air_date") or "")[:4] if show.get("first_air_date") else "N/A"
            text = f"{title} ({year})"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, show_id)
            self.tv_list.addItem(item)

        # =========================
        # People (sorted by popularity)
        # =========================
        people_queries = [q.strip() for q in query.split(",")]
        is_multiple_person_search = len(people_queries) > 1 and all(len(q) > 0 for q in people_queries)

        people = search_tmdb_person(query) if not is_multiple_person_search else []
        people = sorted(people, key=lambda p: p.get("popularity", 0) or 0, reverse=True)
        self.people_list.clear()
        for person in people[:20]:
            name = person.get("name", "N/A")
            dept = person.get("known_for_department", "")
            person_id = person.get("id")
            if dept:
                text = f"{name} ({dept})"
            else:
                text = name
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, person_id)
            self.people_list.addItem(item)

                # =========================
        # Collaborations (multi-person search, sorted by popularity)
        # =========================
        self.collaborations_list.clear()

        if is_multiple_person_search and len(people_queries) >= 2:
            all_people = []

            # For each name in comma-separated query, pick the top TMDb person match.
            for person_query in people_queries:
                persons = search_tmdb_person(person_query)
                if persons:
                    all_people.append((person_query, persons[0]))

            if len(all_people) >= 2:
                person_ids = [p[1].get("id") for p in all_people]
                common_films = find_common_filmography_with_roles(person_ids)

                # Build a list with associated popularity and type (movie/tv).
                enriched_movies = []
                enriched_tv = []

                for film_id, people_roles in common_films:
                    # Try as movie first
                    movie_details = get_tmdb_movie_details(film_id)
                    if movie_details and movie_details.get("title"):
                        pop = movie_details.get("popularity", 0) or 0
                        enriched_movies.append(
                            ("movie", pop, film_id, people_roles, movie_details)
                        )
                    else:
                        # Otherwise, treat as TV show
                        tv_details = get_tmdb_tv_details(film_id)
                        if tv_details and tv_details.get("name"):
                            pop = tv_details.get("popularity", 0) or 0
                            enriched_tv.append(
                                ("tv", pop, film_id, people_roles, tv_details)
                            )

                # Sort each category by popularity descending
                enriched_movies.sort(key=lambda x: x[1], reverse=True)
                enriched_tv.sort(key=lambda x: x[1], reverse=True)

                # Take at most 15 movies and 10 TV shows
                enriched_movies = enriched_movies[:15]
                enriched_tv = enriched_tv[:10]

                # Combine for display: movies first, then TV
                display_enriched = enriched_movies + enriched_tv

                if display_enriched:
                    for media_type, pop, film_id, people_roles, details in display_enriched:
                        if media_type == "movie":
                            title = details.get("title", "Unknown")
                            year = (details.get("release_date") or "")[:4] if details.get("release_date") else "N/A"
                        else:
                            title = details.get("name", "Unknown")
                            year = (details.get("first_air_date") or "")[:4] if details.get("first_air_date") else "N/A"

                        # Build roles string "Name: Role1, Role2 | Other: ..."
                        roles_str = ""
                        for idx, (person_name, _) in enumerate(all_people):
                            if idx in people_roles:
                                roles = ", ".join(people_roles[idx])
                                roles_str += f"{person_name}: {roles} | "
                        roles_str = roles_str.rstrip(" | ")

                        text = f"{title} ({year}) - {roles_str}"
                        item = QListWidgetItem(text)
                        item.setData(Qt.ItemDataRole.UserRole, f"{media_type}:{film_id}")
                        self.collaborations_list.addItem(item)

                    # Tab label shows total count actually displayed
                    self.tabs.setTabText(3, f"🤝 Collaborations ({len(display_enriched)})")
                else:
                    no_collab = QListWidgetItem(
                        f"No collaborations found between {', '.join([p[0] for p in all_people])}"
                    )
                    self.collaborations_list.addItem(no_collab)



    def on_movie_selected(self, item):
        movie_id = item.data(Qt.ItemDataRole.UserRole)
        self.item_selected.emit(f"movie:{movie_id}")
        self.close()

    def on_tv_selected(self, item):
        tv_id = item.data(Qt.ItemDataRole.UserRole)
        self.item_selected.emit(f"tv:{tv_id}")
        self.close()

    def on_people_selected(self, item):
        person_id = item.data(Qt.ItemDataRole.UserRole)
        self.item_selected.emit(f"person:{person_id}")
        self.close()

    def on_collaboration_selected(self, item):
        data = item.data(Qt.ItemDataRole.UserRole)
        self.item_selected.emit(data)
        self.close()


# =================================================
# PERSON DIALOG
# =================================================

class PersonDialog(QDialog):
    item_selected = pyqtSignal(str)

    def __init__(self, person_id, parent=None):
        super().__init__(parent)
        self.person_id = person_id

        self.setWindowTitle("👤 Profile")
        self.resize(1400, 900)
        self.setStyleSheet(STYLESHEET)

        layout = QVBoxLayout(self)

        top_layout = QHBoxLayout()

        self.photo_label = QLabel()
        self.photo_label.setFixedSize(220, 320)
        self.photo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.photo_label.setStyleSheet("border: 2px solid #2a3d52; border-radius: 8px;")
        top_layout.addWidget(self.photo_label)

        self.details_browser = QTextBrowser()
        self.details_browser.setOpenExternalLinks(False)
        self.details_browser.anchorClicked.connect(self.on_link_clicked)
        top_layout.addWidget(self.details_browser)

        layout.addLayout(top_layout)

        person_details = get_tmdb_person_details(person_id)
        filmography = get_tmdb_person_filmography(person_id)

        # Photo
        if person_details and person_details.get("profile_path"):
            img_url = TMDB_IMAGE_BASE + person_details["profile_path"]
            try:
                r = requests.get(img_url, timeout=10)
                if r.status_code == 200:
                    pix = QPixmap()
                    pix.loadFromData(r.content)
                    self.photo_label.setPixmap(
                        pix.scaled(
                            self.photo_label.width(),
                            self.photo_label.height(),
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                    )
                else:
                    self.photo_label.setText("No photo")
            except Exception:
                self.photo_label.setText("No photo")
        else:
            self.photo_label.setText("No photo")

        name = person_details.get("name", "Unknown") if person_details else "Unknown"
        birthday = person_details.get("birthday", "N/A") if person_details else "N/A"
        birthplace = person_details.get("place_of_birth", "N/A") if person_details else "N/A"
        dept = person_details.get("known_for_department", "N/A") if person_details else "N/A"
        biography = person_details.get("biography") if person_details else ""

        cast = filmography.get("cast", [])
        directing = filmography.get("directing", [])
        writing = filmography.get("writing", [])
        producing = filmography.get("producing", [])

        def format_credit_section(title, items, is_cast=False):
            if not items:
                return f"<h3>{title}</h3><p>No entries.</p>"
            parts = [f"<h3>{title}</h3><ul>"]
            for item in items[:20]:
                item_id = item.get("id")
                item_title = item.get("title", "Unknown")
                year = (item.get("release_date") or "")[:4] if item.get("release_date") else ""
                media_type = item.get("type", "movie")
                href = f"{media_type}:{item_id}" if item_id else ""
                extra = ""
                if is_cast and item.get("character"):
                    extra = f" as {item.get('character')}"
                elif not is_cast and item.get("job"):
                    extra = f" ({item.get('job')})"
                year_str = f" ({year})" if year else ""
                if href:
                    parts.append(
                        f'<li><a href="{href}">{item_title}{year_str}</a>{extra}</li>'
                    )
                else:
                    parts.append(f"<li>{item_title}{year_str}{extra}</li>")
            parts.append("</ul>")
            return "".join(parts)

        html_parts = [
            f"<h1>{name}</h1>",
            f"<p><b>Department:</b> {dept}</p>",
            f"<p><b>Born:</b> {birthday} in {birthplace}</p>",
        ]
        if biography:
            html_parts.append(f"<p>{biography}</p>")

        html_parts.append(format_credit_section("Acting Credits", cast, is_cast=True))
        html_parts.append(format_credit_section("Directing Credits", directing, is_cast=False))
        html_parts.append(format_credit_section("Writing Credits", writing, is_cast=False))
        html_parts.append(format_credit_section("Producing Credits", producing, is_cast=False))

        self.details_browser.setHtml("".join(html_parts))

    def on_link_clicked(self, qurl):
        link = qurl.toString()
        if link.startswith("movie:") or link.startswith("tv:") or link.startswith("person:"):
            self.item_selected.emit(link)
            self.close()


# =================================================
# SEASON EPISODES DIALOG (IMDb ratings per episode)
# =================================================

class SeasonEpisodesDialog(QDialog):
    def __init__(self, tmdb_id, season_num, season_name, parent=None):
        super().__init__(parent)
        self.tmdb_id = tmdb_id
        self.season_num = season_num

        self.setWindowTitle(f"📺 {season_name}")
        self.resize(900, 700)
        self.setStyleSheet(STYLESHEET)

        layout = QVBoxLayout(self)

        title_label = QLabel(season_name)
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(title_label)

        self.episodes_browser = QTextBrowser()
        layout.addWidget(self.episodes_browser)

        self.load_episodes()

    def load_episodes(self):
        episodes = get_tmdb_tv_season_episodes(self.tmdb_id, self.season_num)
        if not episodes:
            self.episodes_browser.setHtml("<p>No episodes found for this season.</p>")
            return

        html_parts = []
        for ep in episodes:
            ep_num = ep.get("episode_number", "?")
            name = ep.get("name", "Unknown Episode")
            air_date = ep.get("air_date", "N/A")
            overview = ep.get("overview", "No plot available.")

            # Get IMDb rating for the episode via OMDb
            imdb_rating_str = "N/A"
            try:
                external_ids = get_tmdb_episode_external_ids(self.tmdb_id, self.season_num, ep_num)
                imdb_id = external_ids.get("imdb_id")
                if imdb_id:
                    omdb_data = get_omdb_episode_reviews(imdb_id)
                    if omdb_data:
                        rating = omdb_data.get("imdbRating")
                        if rating and rating != "N/A":
                            imdb_rating_str = rating
                        # Optional: override plot with OMDb plot if present
                        if omdb_data.get("Plot") and omdb_data.get("Plot") != "N/A":
                            overview = omdb_data.get("Plot")
            except Exception:
                pass

            html_parts.append("<hr>")
            html_parts.append(f"<h3>Episode {ep_num}: {name}</h3>")
            html_parts.append(f"<p><b>Air date:</b> {air_date}</p>")
            html_parts.append(f"<p><b>IMDb Rating:</b> {imdb_rating_str}/10</p>")
            html_parts.append(f"<p>{overview}</p>")

        self.episodes_browser.setHtml("".join(html_parts))


# =================================================
# MEDIA DETAILS DIALOG (MOVIE / TV)
# =================================================

class MediaDetailsDialog(QDialog):
    person_selected = pyqtSignal(str)
    item_selected = pyqtSignal(str)

    def __init__(self, media_id, media_type, parent=None):
        super().__init__(parent)
        self.media_id = media_id
        self.media_type = media_type  # "movie" or "tv"

        self.setWindowTitle("🎬 Movie" if media_type == "movie" else "📺 TV Show")
        self.resize(1400, 900)
        self.setStyleSheet(STYLESHEET)

        layout = QVBoxLayout(self)

        top_layout = QHBoxLayout()

        self.poster_label = QLabel()
        self.poster_label.setFixedSize(220, 320)
        self.poster_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.poster_label.setStyleSheet("border: 2px solid #2a3d52; border-radius: 8px;")
        top_layout.addWidget(self.poster_label)

        self.info_browser = QTextBrowser()
        self.info_browser.setOpenExternalLinks(False)
        self.info_browser.anchorClicked.connect(self.on_link_clicked)
        top_layout.addWidget(self.info_browser)

        layout.addLayout(top_layout)

        # Trailer button
        button_layout = QHBoxLayout()
        trailer_btn = QPushButton("🎥 Show Trailer")
        trailer_btn.clicked.connect(self.show_trailer)
        button_layout.addWidget(trailer_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        # For TV: container is used when embedded in main window
        self.episodes_container = QTextBrowser()
        self.episodes_container.setOpenExternalLinks(False)
        self.episodes_container.anchorClicked.connect(self.on_link_clicked)
        self.episodes_container.hide()
        layout.addWidget(self.episodes_container)

        self.tv_details_cache = None
        self.videos = []

        if media_type == "movie":
            self.load_movie()
        else:
            self.load_tv()

    def _load_poster(self, details):
        if details.get("poster_path"):
            img_url = TMDB_IMAGE_LARGE + details["poster_path"]
            try:
                r = requests.get(img_url, timeout=10)
                if r.status_code == 200:
                    pix = QPixmap()
                    pix.loadFromData(r.content)
                    self.poster_label.setPixmap(
                        pix.scaled(
                            self.poster_label.width(),
                            self.poster_label.height(),
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                    )
                    return
            except Exception:
                pass
        self.poster_label.setText("No poster")

    def load_movie(self):
        details = get_tmdb_movie_details(self.media_id)
        credits = get_tmdb_movie_credits(self.media_id)
        self.videos = get_tmdb_movie_videos(self.media_id)

        title = details.get("title", "Unknown")

        # IMDb rating via OMDb using IMDb ID
        imdb_rating = "N/A"
        try:
            external_ids = get_tmdb_external_ids(self.media_id, "movie")
            imdb_id = external_ids.get("imdb_id")
            if imdb_id:
                omdb_data = get_omdb_by_imdb_id(imdb_id)
                if omdb_data and omdb_data.get("imdbRating") and omdb_data.get("imdbRating") != "N/A":
                    imdb_rating = omdb_data.get("imdbRating")
        except Exception:
            pass

        self._load_poster(details)

        year = (details.get("release_date") or "")[:4] if details.get("release_date") else "N/A"
        runtime = details.get("runtime", "N/A")
        genres = ", ".join(g.get("name", "") for g in details.get("genres", []))
        plot = details.get("overview", "N/A")

        # Box Office Data (Budget and Worldwide Gross)
        budget = format_currency(details.get("budget", 0))
        worldwide_gross = format_currency(details.get("revenue", 0))

        # Directors
        directors_html = []
        for crew in credits.get("crew", []) or []:
            if crew.get("job") == "Director":
                name = crew.get("name", "Unknown")
                pid = crew.get("id")
                if pid:
                    directors_html.append(f'<a href="person:{pid}">{name}</a>')
                else:
                    directors_html.append(name)
        directors_html_str = ", ".join(directors_html) if directors_html else "N/A"

        # Cast
        cast_html = []
        for cast in (credits.get("cast", []) or [])[:20]:
            name = cast.get("name", "Unknown")
            character = cast.get("character", "")
            pid = cast.get("id")
            label = name
            if character:
                label += f" as {character}"
            if pid:
                cast_html.append(f'<li><a href="person:{pid}">{label}</a></li>')
            else:
                cast_html.append(f"<li>{label}</li>")

        cast_html_str = "<ul>" + "".join(cast_html) + "</ul>" if cast_html else "N/A"

        # Creators
        creators_html = []
        for creator in details.get("created_by", []) or []:
            name = creator.get("name", "Unknown")
            cid = creator.get("id")
            if cid:
                creators_html.append(f'<li><a href="person:{cid}">{name}</a></li>')
            else:
                creators_html.append(f"<li>{name}</li>")
        creators_html_str = "<ul>" + "".join(creators_html) + "</ul>" if creators_html else "N/A"

        info_html = f"""
        <h1>{title} ({year})</h1>
        <p><b>IMDb Rating:</b> {imdb_rating}/10</p>
        <p><b>Runtime:</b> {runtime} min</p>
        <p><b>Genres:</b> {genres}</p>
        <p><b>Budget:</b> {budget}</p>
        <p><b>Worldwide Gross:</b> {worldwide_gross}</p>
        <p><b>Plot:</b><br>{plot}</p>
        <h3>Directors</h3>
        <p>{directors_html_str}</p>
        <h3>Cast</h3>
        {cast_html_str}
        <h3>Creators</h3>
        {creators_html_str}
        """

        self.info_browser.setHtml(info_html)

    def load_tv(self):
        details = get_tmdb_tv_details(self.media_id)
        credits = get_tmdb_tv_credits(self.media_id)
        self.videos = get_tmdb_tv_videos(self.media_id)
        self.tv_details_cache = details

        title = details.get("name", "Unknown")

        imdb_rating = "N/A"
        try:
            external_ids = get_tmdb_external_ids(self.media_id, "tv")
            imdb_id = external_ids.get("imdb_id")
            if imdb_id:
                omdb_data = get_omdb_by_imdb_id(imdb_id)
                if omdb_data and omdb_data.get("imdbRating") and omdb_data.get("imdbRating") != "N/A":
                    imdb_rating = omdb_data.get("imdbRating")
        except Exception:
            pass

        self._load_poster(details)

        year = (details.get("first_air_date") or "")[:4] if details.get("first_air_date") else "N/A"
        genres = ", ".join(g.get("name", "") for g in details.get("genres", []))
        plot = details.get("overview", "N/A")
        num_seasons = details.get("number_of_seasons", "N/A")
        num_episodes = details.get("number_of_episodes", "N/A")
        networks = ", ".join(n.get("name", "") for n in details.get("networks", []))
        status = details.get("status", "N/A")

        # Seasons list (clickable)
        seasons_html_parts = []
        for season in details.get("seasons", []) or []:
            sn = season.get("season_number")
            if sn == 0:
                # Skip "Specials"
                continue
            sname = season.get("name") or f"Season {sn}"
            airdate = season.get("air_date") or ""
            year_s = airdate[:4] if airdate else ""
            epcount = season.get("episode_count")
            label = sname
            if year_s:
                label += f" ({year_s})"
            if epcount:
                label += f" - {epcount} episodes"
            seasons_html_parts.append(f'<li><a href="season:{sn}">{label}</a></li>')

        seasons_html_str = "<ul>" + "".join(seasons_html_parts) + "</ul>" if seasons_html_parts else "N/A"

        # Creators
        creators_html = []
        for creator in details.get("created_by", []) or []:
            name = creator.get("name", "Unknown")
            cid = creator.get("id")
            if cid:
                creators_html.append(f'<li><a href="person:{cid}">{name}</a></li>')
            else:
                creators_html.append(f"<li>{name}</li>")
        creators_html_str = "<ul>" + "".join(creators_html) + "</ul>" if creators_html else "N/A"


        # Cast
        cast_html = []
        for cast in (credits.get("cast", []) or [])[:20]:
            name = cast.get("name", "Unknown")
            character = cast.get("character", "")
            pid = cast.get("id")
            label = name
            if character:
                label += f" as {character}"
            if pid:
                cast_html.append(f'<li><a href="person:{pid}">{label}</a></li>')
            else:
                cast_html.append(f"<li>{label}</li>")
        cast_html_str = "<ul>" + "".join(cast_html) + "</ul>" if cast_html else "N/A"

        info_html = f"""
        <h1>{title} ({year})</h1>
        <p><b>IMDb Rating:</b> {imdb_rating}/10</p>
        <p><b>Status:</b> {status}</p>
        <p><b>Genres:</b> {genres}</p>
        <p><b>Seasons:</b> {num_seasons} | <b>Episodes:</b> {num_episodes}</p>
        <p><b>Networks:</b> {networks}</p>
        <p><b>Plot:</b><br>{plot}</p>
        <h3>Seasons</h3>
        {seasons_html_str}
        <h3>Cast</h3>
        {cast_html_str}
        <h3>Creators</h3>
        {creators_html_str}
        """

        self.info_browser.setHtml(info_html)

    def on_link_clicked(self, qurl):
        link = qurl.toString()
        if not link:
            return

        if link.startswith("season:"):
            try:
                season_num = int(link.split(":")[1])
            except Exception:
                return

            details = self.tv_details_cache or get_tmdb_tv_details(self.media_id)
            self.tv_details_cache = details

            season_name = f"Season {season_num}"
            for season in details.get("seasons", []) or []:
                if season.get("season_number") == season_num:
                    season_name = season.get("name", season_name)
                    break

            # Open dedicated window for this season's episodes, then reload TV page
            episodes_dialog = SeasonEpisodesDialog(self.media_id, season_num, season_name, self)
            episodes_dialog.exec()
            self.load_tv()
            return

        if link.startswith("person:"):
            self.person_selected.emit(link)
            return

        if link.startswith("movie:") or link.startswith("tv:"):
            self.item_selected.emit(link)
            return

    def show_trailer(self):
        if not self.videos:
            QMessageBox.information(self, "No Trailer", "No videos found for this content.")
            return

        trailer = None
        for video in self.videos:
            if video.get("type") == "Trailer" and video.get("site") == "YouTube":
                trailer = video
                break

        if not trailer:
            for video in self.videos:
                if video.get("site") == "YouTube":
                    trailer = video
                    break

        if trailer and trailer.get("key"):
            webbrowser.open(f"https://www.youtube.com/watch?v={trailer.get('key')}")
        else:
            QMessageBox.information(self, "No Trailer", "No YouTube trailer found.")


# =================================================
# MAIN WINDOW
# =================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("🎬 Movie/TV Browser")
        self.setGeometry(100, 100, 1600, 900)
        self.setStyleSheet(STYLESHEET)

        self.current_media_id = None
        self.current_media_type = None
        self.current_details_dialog = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Top button bar
        button_layout = QHBoxLayout()

        search_btn = QPushButton("🔍 Search")
        search_btn.clicked.connect(self.open_search)
        button_layout.addWidget(search_btn)

        voice_btn = QPushButton("🎤 Voice Search")
        voice_btn.clicked.connect(self.open_voice)
        button_layout.addWidget(voice_btn)

        button_layout.addStretch()

        self.home_btn = QPushButton("🏠 Home")
        self.home_btn.clicked.connect(self.show_home)
        self.home_btn.setVisible(False)
        button_layout.addWidget(self.home_btn)

        layout.addLayout(button_layout)

        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_widget)
        layout.addWidget(self.main_widget)

        self.search_window = None
        self.voice_window = None

        self.show_home()

    def clear_main_layout(self):
        while self.main_layout.count():
            item = self.main_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def show_home(self):
        """
        Show trending recommendations (no TMDb scores displayed).
        """
        self.current_media_id = None
        self.current_media_type = None
        self.home_btn.setVisible(False)

        self.clear_main_layout()

        main_scroll = QScrollArea()
        main_scroll.setWidgetResizable(True)
        main_content = QWidget()
        main_content_layout = QVBoxLayout(main_content)

        # Movies section
        movies_label = QLabel("🔥 Trending Movies")
        movies_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        main_content_layout.addWidget(movies_label)

        movies_scroll = QScrollArea()
        movies_scroll.setWidgetResizable(True)
        movies_scroll.setFixedHeight(350)
        movies_grid = QWidget()
        movies_layout = QGridLayout(movies_grid)
        movies_layout.setSpacing(15)
        movies_scroll.setWidget(movies_grid)
        main_content_layout.addWidget(movies_scroll)

        # TV section
        tv_label = QLabel("📺 Trending TV Shows")
        tv_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        main_content_layout.addWidget(tv_label)

        tv_scroll = QScrollArea()
        tv_scroll.setWidgetResizable(True)
        tv_scroll.setFixedHeight(350)
        tv_grid = QWidget()
        tv_layout = QGridLayout(tv_grid)
        tv_layout.setSpacing(15)
        tv_scroll.setWidget(tv_grid)
        main_content_layout.addWidget(tv_scroll)

        main_content_layout.addStretch()
        main_scroll.setWidget(main_content)
        self.main_layout.addWidget(main_scroll)

        movies = get_tmdb_recommendations("movie", 10)
        tv_shows = get_tmdb_recommendations("tv", 10)

        for col, movie in enumerate(movies):
            self.add_poster_button(movies_layout, movie, "movie", col)

        for col, show in enumerate(tv_shows):
            self.add_poster_button(tv_layout, show, "tv", col)

    def add_poster_button(self, layout, item, media_type, col):
        # Basic metadata for the tile
        item_id = item.get("id")
        title = item.get("title") if media_type == "movie" else item.get("name")
        year = (
            (item.get("release_date") or "")[:4]
            if media_type == "movie"
            else (item.get("first_air_date") or "")[:4]
        )

        # Try to get poster_path from the list item first
        poster_path = item.get("poster_path")

        # If the trending/list result has no poster_path, fall back to full details
        if not poster_path and item_id:
            try:
                if media_type == "movie":
                    details = get_tmdb_movie_details(item_id)
                else:
                    details = get_tmdb_tv_details(item_id)
                poster_path = details.get("poster_path")
            except Exception:
                poster_path = None

        btn = QPushButton()
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setFixedSize(200, 300)

        if poster_path:
            # Try small size first, then large; longer timeout to match detail page
            img_urls = [
                TMDB_IMAGE_BASE + poster_path,
                TMDB_IMAGE_LARGE + poster_path,
            ]
            loaded = False
            for img_url in img_urls:
                try:
                    r = requests.get(img_url, timeout=10)
                    if r.status_code == 200:
                        pix = QPixmap()
                        pix.loadFromData(r.content)
                        btn.setIcon(QIcon(pix))
                        btn.setIconSize(QSize(200, 300))
                        loaded = True
                        break
                except Exception:
                    continue

            if not loaded:
                # If all image attempts fail, show title/year text
                btn.setText(f"{title}\n({year})")
        else:
            # No poster path even after detail lookup
            btn.setText(f"{title}\n({year})")

        btn.clicked.connect(
            lambda checked, mid=item_id, mtype=media_type: self.show_media_details(mid, mtype)
        )
        layout.addWidget(btn, 0, col)


    def open_search(self):
        if self.search_window is None:
            self.search_window = SearchWindow(self)
            self.search_window.item_selected.connect(self.on_item_selected)
        self.search_window.show()
        self.search_window.raise_()
        self.search_window.activateWindow()

    def open_voice(self):
        if self.voice_window is None:
            self.voice_window = VoiceRecordWindow(self)
            self.voice_window.search_requested.connect(self.on_voice_search)
        self.voice_window.show()
        self.voice_window.raise_()
        self.voice_window.activateWindow()

    def on_voice_search(self, text):
        if self.search_window is None:
            self.search_window = SearchWindow(self)
            self.search_window.item_selected.connect(self.on_item_selected)
        self.search_window.search_input.setText(text)
        self.search_window.show()
        self.search_window.raise_()
        self.search_window.activateWindow()
        self.search_window.on_search_button_clicked()

    def on_item_selected(self, item_str):
        if not item_str:
            return
        parts = item_str.split(":")
        if len(parts) != 2:
            return
        type_, id_ = parts[0], parts[1]
        if type_ == "person":
            self.show_person_details(int(id_))
        elif type_ in ("movie", "tv"):
            self.show_media_details(int(id_), type_)

    def show_person_details(self, person_id):
        dialog = PersonDialog(person_id, self)
        dialog.item_selected.connect(self.on_item_selected)
        dialog.exec()

        # When person dialog closes, reload the movie/TV page that was open
        if self.current_media_id is not None and self.current_media_type is not None:
            self.show_media_details(self.current_media_id, self.current_media_type)

    def show_media_details(self, media_id, media_type):
        self.current_media_id = media_id
        self.current_media_type = media_type
        self.home_btn.setVisible(True)

        self.clear_main_layout()

        dialog = MediaDetailsDialog(media_id, media_type, self)
        dialog.item_selected.connect(self.on_item_selected)
        dialog.person_selected.connect(self.on_item_selected)

        self.show_details_in_main(dialog)

    def show_details_in_main(self, dialog):
        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)

        # Top layout with poster and info
        top_layout = QHBoxLayout()
        top_layout.addWidget(dialog.poster_label)
        top_layout.addWidget(dialog.info_browser)
        details_layout.addLayout(top_layout)

        # Trailer button
        btn_layout = QHBoxLayout()
        trailer_btn = QPushButton("🎥 Show Trailer")
        trailer_btn.clicked.connect(dialog.show_trailer)
        btn_layout.addWidget(trailer_btn)
        btn_layout.addStretch()
        details_layout.addLayout(btn_layout)

        # Episodes container (for TV)
        details_layout.addWidget(dialog.episodes_container)

        self.main_layout.addWidget(details_widget)
        self.current_details_dialog = dialog


# =================================================
# MAIN
# =================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Set Apple-style font globally with fallback
    font = QFont("SF Pro Text", 14)
    if not QFontInfo(font).family():  # font not found
        font = QFont("Helvetica Neue", 14)
    app.setFont(font)

    actual_font = QFontInfo(font).family()
    print(f"Using the Font: {actual_font}")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
