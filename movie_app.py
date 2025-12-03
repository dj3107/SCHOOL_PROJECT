import wx
import wx.lib.scrolledpanel as scrolled
import requests
import threading
import time
import numpy as np
import whisper
import sounddevice as sd
from dataclasses import dataclass
from typing import List, Dict, Optional

# =====================================================
# API KEYS
# =====================================================
OMDB_KEY = "cc0fba9e"
TMDB_KEY = "b3a56cb04a5f02d1229c0fc11b3303c2"

# =====================================================
# DATA CLASSES
# =====================================================
@dataclass
class MovieData:
    imdb_id: str
    title: str
    year: str
    poster_url: str
    rating: float
    genre: str
    runtime: str
    director: str
    actors: str
    plot: str
    budget: str
    gross: str

@dataclass
class PersonData:
    tmdb_id: int
    name: str
    profile_path: str
    biography: str
    birthday: str
    deathday: str
    birthplace: str
    known_for: str

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def search_omdb(title, type_, api_key=OMDB_KEY):
    """Generic OMDb search."""
    if not title:
        return []
    url = "http://www.omdbapi.com/"
    params = {"apikey": api_key, "s": title, "type": type_}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("Response") == "True":
            return data.get("Search", [])
        else:
            print("OMDb:", data.get("Error"))
            return []
    except requests.RequestException as e:
        print("Network/API error (OMDb):", e)
        return []

def search_movies(movie_title, api_key=OMDB_KEY):
    return search_omdb(movie_title, "movie", api_key=api_key)

def search_tv_series(series_title, api_key=OMDB_KEY):
    return search_omdb(series_title, "series", api_key=api_key)

def get_title_by_id(imdb_id, api_key=OMDB_KEY):
    """Get full title details from OMDb by IMDb ID."""
    url = "http://www.omdbapi.com/"
    params = {"apikey": api_key, "i": imdb_id, "plot": "full"}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("Response") == "True":
            return data
        else:
            print("OMDb:", data.get("Error"))
            return None
    except requests.RequestException as e:
        print("Network/API error (OMDb):", e)
        return None

def search_tmdb_movie_for_id(title, api_key=TMDB_KEY):
    try:
        url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": api_key, "query": title, "page": 1}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        results = data.get("results") or []
        return results[0]["id"] if results else None
    except requests.RequestException as e:
        print("TMDb movie search error:", e)
        return None

def search_tmdb_tv_for_id(name, api_key=TMDB_KEY):
    try:
        url = "https://api.themoviedb.org/3/search/tv"
        params = {"api_key": api_key, "query": name, "page": 1}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        results = data.get("results") or []
        return results[0]["id"] if results else None
    except requests.RequestException as e:
        print("TMDb TV search error:", e)
        return None

def search_tmdb_person(name, api_key=TMDB_KEY):
    """Search TMDb for people."""
    try:
        url = "https://api.themoviedb.org/3/search/person"
        params = {"api_key": api_key, "query": name, "page": 1}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get("results", [])
    except requests.RequestException as e:
        print("TMDb person search error:", e)
        return []

def get_tmdb_person_details(person_id, api_key=TMDB_KEY):
    """Get TMDb person details."""
    if not person_id:
        return None
    try:
        url = f"https://api.themoviedb.org/3/person/{person_id}"
        params = {"api_key": api_key, "append_to_response": "combined_credits"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print("TMDb person details error:", e)
        return None

def get_tmdb_movie_details(tmdb_movie_id, api_key=TMDB_KEY):
    if not tmdb_movie_id:
        return {"budget": "N/A", "revenue": "N/A"}
    try:
        url = f"https://api.themoviedb.org/3/movie/{tmdb_movie_id}"
        params = {"api_key": api_key, "append_to_response": "credits"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print("TMDb movie details error:", e)
        return {}

def get_tmdb_tv_details(tmdb_tv_id, api_key=TMDB_KEY):
    """Get TV show info from TMDb."""
    if not tmdb_tv_id:
        return {}
    try:
        url = f"https://api.themoviedb.org/3/tv/{tmdb_tv_id}"
        params = {"api_key": api_key, "append_to_response": "credits,seasons"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print("TMDb TV details error:", e)
        return {}

def get_tmdb_season_episodes(tv_id, season_num, api_key=TMDB_KEY):
    """Get episodes for a TV season."""
    try:
        url = f"https://api.themoviedb.org/3/tv/{tv_id}/season/{season_num}"
        params = {"api_key": api_key}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("episodes", [])
    except requests.RequestException as e:
        print("TMDb season episodes error:", e)
        return []

def get_tmdb_person_filmography(person_id, api_key=TMDB_KEY):
    """Get person's filmography."""
    try:
        url = f"https://api.themoviedb.org/3/person/{person_id}/combined_credits"
        params = {"api_key": api_key}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        cast = data.get("cast", [])
        return sorted(cast, key=lambda x: x.get("popularity", 0), reverse=True)[:20]
    except requests.RequestException as e:
        print("TMDb filmography error:", e)
        return []

def format_currency(amount):
    if amount and amount > 0:
        return f"${amount:,.2f}"
    return "N/A"

def remove_duplicates(movies):
    seen = set()
    unique = []
    for m in movies:
        imdb = m.get("imdbID")
        if imdb and imdb not in seen:
            seen.add(imdb)
            unique.append(m)
    return unique

# =====================================================
# WHISPER RECORDING THREAD
# =====================================================
class WhisperThread(threading.Thread):
    def __init__(self, callback, error_callback, model_name="medium"):
        super().__init__(daemon=True)
        self.callback = callback
        self.error_callback = error_callback
        self.model_name = model_name
        self.running = False
        self.recorded_audio = []
        try:
            self.model = whisper.load_model(self.model_name)
        except Exception as e:
            self.model = None
            self.error_callback(f"Could not load Whisper model: {e}")

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
            self.error_callback(str(e))
            return

        if len(self.recorded_audio) == 0:
            self.error_callback("No audio recorded.")
            return

        audio_np = np.concatenate([a[:, 0] for a in self.recorded_audio]).astype(np.float32) / 32768.0

        if np.max(np.abs(audio_np)) < 0.01:
            self.error_callback("Too quiet. No speech detected.")
            return

        try:
            result = self.model.transcribe(audio_np, fp16=False)
            text = result.get("text", "").strip()
            if not text:
                self.error_callback("Could not recognize speech.")
                return
            self.callback(text)
        except Exception as e:
            self.error_callback(str(e))

    def stop(self):
        self.running = False

# =====================================================
# MAIN APPLICATION
# =====================================================
class MovieApp(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Movie/TV/Person Search", size=(1600, 900))
        
        self.whisper_thread = None
        
        # Create notebook (tabs)
        self.notebook = wx.Notebook(self)
        
        # Create tabs
        self.movie_panel = wx.Panel(self.notebook)
        self.tv_panel = wx.Panel(self.notebook)
        self.person_panel = wx.Panel(self.notebook)
        
        self.notebook.AddPage(self.movie_panel, "🎬 Movies")
        self.notebook.AddPage(self.tv_panel, "📺 TV Shows")
        self.notebook.AddPage(self.person_panel, "👤 People")
        
        # Setup tabs
        self.setup_movie_tab()
        self.setup_tv_tab()
        self.setup_person_tab()
        
        self.Centre()
        self.Show()

    # ==================== MOVIE TAB ====================
    def setup_movie_tab(self):
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Left panel
        left_sizer = wx.BoxSizer(wx.VERTICAL)
        
        self.movie_search = wx.TextCtrl(self.movie_panel, value="", size=(300, -1))
        self.movie_search.SetHint("Search movies...")
        left_sizer.Add(self.movie_search, 0, wx.ALL | wx.EXPAND, 5)
        
        # Button row
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        search_btn = wx.Button(self.movie_panel, label="🔍 Search")
        search_btn.Bind(wx.EVT_BUTTON, self.search_movies)
        btn_sizer.Add(search_btn, 0, wx.ALL, 5)
        
        self.movie_record_btn = wx.Button(self.movie_panel, label="🎤 Record")
        self.movie_record_btn.Bind(wx.EVT_BUTTON, self.start_record_movie)
        btn_sizer.Add(self.movie_record_btn, 0, wx.ALL, 5)
        
        self.movie_stop_btn = wx.Button(self.movie_panel, label="🛑 Stop")
        self.movie_stop_btn.Bind(wx.EVT_BUTTON, self.stop_record_movie)
        self.movie_stop_btn.Enable(False)
        btn_sizer.Add(self.movie_stop_btn, 0, wx.ALL, 5)
        
        left_sizer.Add(btn_sizer, 0, wx.EXPAND)
        
        # Results list
        self.movie_list_ctrl = wx.ListCtrl(self.movie_panel, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        self.movie_list_ctrl.AppendColumn("Title", width=280)
        self.movie_list_ctrl.Bind(wx.EVT_LIST_ITEM_SELECTED, self.select_movie)
        left_sizer.Add(self.movie_list_ctrl, 1, wx.ALL | wx.EXPAND, 5)
        
        # Right panel
        right_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Poster
        self.movie_poster = wx.StaticBitmap(self.movie_panel)
        self.movie_poster.SetMinSize((280, 420))
        right_sizer.Add(self.movie_poster, 0, wx.ALL | wx.CENTER, 5)
        
        # Details
        self.movie_details = wx.TextCtrl(
            self.movie_panel, 
            value="", 
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_WORDWRAP
        )
        right_sizer.Add(self.movie_details, 1, wx.ALL | wx.EXPAND, 5)
        
        sizer.Add(left_sizer, 1, wx.EXPAND)
        sizer.Add(right_sizer, 1, wx.EXPAND)
        
        self.movie_panel.SetSizer(sizer)
        self.movie_list = []

    def search_movies(self, event=None):
        title = self.movie_search.GetValue().strip()
        if not title:
            return
        
        results = search_movies(title)
        self.movie_list = remove_duplicates(results)
        
        self.movie_list_ctrl.DeleteAllItems()
        for i, movie in enumerate(self.movie_list):
            self.movie_list_ctrl.InsertItem(i, f"{movie['Title']} ({movie['Year']})")

    def select_movie(self, event):
        index = event.GetIndex()
        if index < 0 or index >= len(self.movie_list):
            return
        
        movie = self.movie_list[index]
        imdb_id = movie["imdbID"]
        
        info = get_title_by_id(imdb_id)
        if not info:
            return
        
        tmdb_id = search_tmdb_movie_for_id(info["Title"])
        tmdb_details = get_tmdb_movie_details(tmdb_id)
        
        # Build details text
        budget = format_currency(tmdb_details.get("budget", 0))
        revenue = format_currency(tmdb_details.get("revenue", 0))
        
        # Get cast
        cast_text = "N/A"
        if "credits" in tmdb_details and tmdb_details["credits"].get("cast"):
            cast_list = tmdb_details["credits"]["cast"][:5]
            cast_text = ", ".join([f"{c['name']}" for c in cast_list])
        
        details_text = f"""Title: {info['Title']}
Year: {info['Year']}
IMDb Rating: {info['imdbRating']} / 10
Genre: {info['Genre']}
Runtime: {info['Runtime']}
Director: {info['Director']}
Cast: {cast_text}

Plot:
{info['Plot']}

Budget: {budget}
Worldwide Gross: {revenue}"""
        
        self.movie_details.SetValue(details_text)
        
        # Load poster
        poster_url = info.get("Poster")
        if poster_url and poster_url != "N/A":
            try:
                response = requests.get(poster_url, timeout=10)
                image_data = response.content
                image = wx.Image(wx.BytesIO(image_data), wx.BITMAP_TYPE_JPEG)
                image.Rescale(280, 420, wx.IMAGE_QUALITY_HIGH)
                self.movie_poster.SetBitmap(wx.Bitmap(image))
            except Exception as e:
                print(f"Error loading poster: {e}")

    def start_record_movie(self, event):
        self.movie_details.SetValue("🎤 Recording... Speak now.\nPress STOP when done.")
        self.movie_record_btn.Enable(False)
        self.movie_stop_btn.Enable(True)
        
        self.whisper_thread = WhisperThread(
            self.voice_done_movie,
            self.voice_error_movie,
            model_name="medium"
        )
        self.whisper_thread.start()

    def stop_record_movie(self, event):
        if self.whisper_thread:
            self.whisper_thread.stop()
        self.movie_stop_btn.Enable(False)
        self.movie_details.SetValue("⏳ Processing speech...")

    def voice_done_movie(self, text):
        wx.CallAfter(self._voice_done_movie_ui, text)

    def _voice_done_movie_ui(self, text):
        self.movie_record_btn.Enable(True)
        self.movie_stop_btn.Enable(False)
        self.movie_search.SetValue(text)
        self.search_movies()

    def voice_error_movie(self, msg):
        wx.CallAfter(self._voice_error_ui, msg, "Movie Search")

    # ==================== TV TAB ====================
    def setup_tv_tab(self):
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Left panel
        left_sizer = wx.BoxSizer(wx.VERTICAL)
        
        self.tv_search = wx.TextCtrl(self.tv_panel, value="", size=(300, -1))
        self.tv_search.SetHint("Search TV shows...")
        left_sizer.Add(self.tv_search, 0, wx.ALL | wx.EXPAND, 5)
        
        # Button row
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        search_btn = wx.Button(self.tv_panel, label="🔍 Search")
        search_btn.Bind(wx.EVT_BUTTON, self.search_tv)
        btn_sizer.Add(search_btn, 0, wx.ALL, 5)
        
        self.tv_record_btn = wx.Button(self.tv_panel, label="🎤 Record")
        self.tv_record_btn.Bind(wx.EVT_BUTTON, self.start_record_tv)
        btn_sizer.Add(self.tv_record_btn, 0, wx.ALL, 5)
        
        self.tv_stop_btn = wx.Button(self.tv_panel, label="🛑 Stop")
        self.tv_stop_btn.Bind(wx.EVT_BUTTON, self.stop_record_tv)
        self.tv_stop_btn.Enable(False)
        btn_sizer.Add(self.tv_stop_btn, 0, wx.ALL, 5)
        
        left_sizer.Add(btn_sizer, 0, wx.EXPAND)
        
        # Results list
        self.tv_list_ctrl = wx.ListCtrl(self.tv_panel, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        self.tv_list_ctrl.AppendColumn("Title", width=280)
        self.tv_list_ctrl.Bind(wx.EVT_LIST_ITEM_SELECTED, self.select_tv)
        left_sizer.Add(self.tv_list_ctrl, 1, wx.ALL | wx.EXPAND, 5)
        
        # Right panel
        right_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Poster
        self.tv_poster = wx.StaticBitmap(self.tv_panel)
        self.tv_poster.SetMinSize((280, 420))
        right_sizer.Add(self.tv_poster, 0, wx.ALL | wx.CENTER, 5)
        
        # Details in scrolled panel
        self.tv_details_scroll = scrolled.ScrolledPanel(self.tv_panel, size=(400, 300))
        self.tv_details_scroll.SetupScrolling()
        self.tv_details_sizer = wx.BoxSizer(wx.VERTICAL)
        self.tv_details_scroll.SetSizer(self.tv_details_sizer)
        right_sizer.Add(self.tv_details_scroll, 1, wx.ALL | wx.EXPAND, 5)
        
        sizer.Add(left_sizer, 1, wx.EXPAND)
        sizer.Add(right_sizer, 1, wx.EXPAND)
        
        self.tv_panel.SetSizer(sizer)
        self.tv_list = []

    def search_tv(self, event=None):
        title = self.tv_search.GetValue().strip()
        if not title:
            return
        
        results = search_tv_series(title)
        self.tv_list = remove_duplicates(results)
        
        self.tv_list_ctrl.DeleteAllItems()
        for i, show in enumerate(self.tv_list):
            self.tv_list_ctrl.InsertItem(i, f"{show['Title']} ({show['Year']})")

    def select_tv(self, event):
        index = event.GetIndex()
        if index < 0 or index >= len(self.tv_list):
            return
        
        tv = self.tv_list[index]
        imdb_id = tv["imdbID"]
        
        info = get_title_by_id(imdb_id)
        if not info:
            return
        
        tmdb_id = search_tmdb_tv_for_id(info["Title"])
        tmdb_details = get_tmdb_tv_details(tmdb_id)
        
        # Clear previous widgets
        self.tv_details_sizer.Clear(True)
        
        # Title and basic info
        title_text = wx.StaticText(self.tv_details_scroll, label=f"Title: {info['Title']}")
        title_font = title_text.GetFont()
        title_font.PointSize += 2
        title_font = title_font.Bold()
        title_text.SetFont(title_font)
        self.tv_details_sizer.Add(title_text, 0, wx.ALL, 5)
        
        basic_info = f"""Year: {info['Year']}
IMDb Rating: {info['imdbRating']} / 10
Genre: {info['Genre']}
Runtime: {info['Runtime']}

Plot:
{info['Plot']}"""
        
        info_text = wx.StaticText(self.tv_details_scroll, label=basic_info)
        self.tv_details_sizer.Add(info_text, 0, wx.ALL | wx.EXPAND, 5)
        
        # Episodes by season
        if tmdb_details and tmdb_details.get("seasons"):
            sep = wx.StaticLine(self.tv_details_scroll)
            self.tv_details_sizer.Add(sep, 0, wx.EXPAND | wx.ALL, 5)
            
            seasons_label = wx.StaticText(self.tv_details_scroll, label="Episodes:")
            seasons_font = seasons_label.GetFont()
            seasons_font = seasons_font.Bold()
            seasons_label.SetFont(seasons_font)
            self.tv_details_sizer.Add(seasons_label, 0, wx.ALL, 5)
            
            for season in tmdb_details.get("seasons", []):
                season_num = season.get("season_number")
                if season_num is None or season_num == 0:
                    continue
                
                season_btn = wx.Button(
                    self.tv_details_scroll,
                    label=f"Season {season_num}"
                )
                season_btn.Bind(
                    wx.EVT_BUTTON,
                    lambda e, s=season_num, tv_id=tmdb_id: self.show_season_episodes(e, s, tv_id)
                )
                self.tv_details_sizer.Add(season_btn, 0, wx.ALL | wx.EXPAND, 3)
        
        self.tv_details_scroll.SetupScrolling()
        
        # Load poster
        poster_url = info.get("Poster")
        if poster_url and poster_url != "N/A":
            try:
                response = requests.get(poster_url, timeout=10)
                image_data = response.content
                image = wx.Image(wx.BytesIO(image_data), wx.BITMAP_TYPE_JPEG)
                image.Rescale(280, 420, wx.IMAGE_QUALITY_HIGH)
                self.tv_poster.SetBitmap(wx.Bitmap(image))
            except Exception as e:
                print(f"Error loading poster: {e}")

    def show_season_episodes(self, event, season_num, tv_id):
        """Display episodes for a season."""
        episodes = get_tmdb_season_episodes(tv_id, season_num)
        
        if not episodes:
            wx.MessageBox("No episodes found.", "Info")
            return
        
        # Create a new frame to show episodes
        frame = wx.Frame(None, title=f"Season {season_num} Episodes")
        panel = wx.Panel(frame)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        scroll = scrolled.ScrolledPanel(panel)
        scroll.SetupScrolling()
        scroll_sizer = wx.BoxSizer(wx.VERTICAL)
        
        for ep in episodes:
            ep_title = ep.get("name", "Unknown")
            ep_num = ep.get("episode_number", "?")
            ep_plot = ep.get("overview", "No plot available.")
            air_date = ep.get("air_date", "N/A")
            
            # Episode header (clickable for details)
            ep_header = wx.StaticText(
                scroll,
                label=f"Episode {ep_num}: {ep_title} ({air_date})"
            )
            ep_font = ep_header.GetFont()
            ep_font = ep_font.Bold()
            ep_header.SetFont(ep_font)
            ep_header.SetForegroundColour(wx.Colour(0, 0, 200))
            scroll_sizer.Add(ep_header, 0, wx.ALL | wx.EXPAND, 5)
            
            ep_plot_text = wx.StaticText(scroll, label=ep_plot)
            ep_plot_text.Wrap(600)
            scroll_sizer.Add(ep_plot_text, 0, wx.ALL | wx.EXPAND, 10)
            
            sep = wx.StaticLine(scroll)
            scroll_sizer.Add(sep, 0, wx.EXPAND | wx.ALL, 5)
        
        scroll.SetSizer(scroll_sizer)
        sizer.Add(scroll, 1, wx.EXPAND)
        panel.SetSizer(sizer)
        
        frame.SetSize(700, 600)
        frame.Show()

    def start_record_tv(self, event):
        self.tv_details_scroll.GetParent().Refresh()
        self.tv_record_btn.Enable(False)
        self.tv_stop_btn.Enable(True)
        
        self.whisper_thread = WhisperThread(
            self.voice_done_tv,
            self.voice_error_tv,
            model_name="medium"
        )
        self.whisper_thread.start()

    def stop_record_tv(self, event):
        if self.whisper_thread:
            self.whisper_thread.stop()
        self.tv_stop_btn.Enable(False)

    def voice_done_tv(self, text):
        wx.CallAfter(self._voice_done_tv_ui, text)

    def _voice_done_tv_ui(self, text):
        self.tv_record_btn.Enable(True)
        self.tv_stop_btn.Enable(False)
        self.tv_search.SetValue(text)
        self.search_tv()

    def voice_error_tv(self, msg):
        wx.CallAfter(self._voice_error_ui, msg, "TV Search")

    # ==================== PERSON TAB ====================
    def setup_person_tab(self):
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Left panel
        left_sizer = wx.BoxSizer(wx.VERTICAL)
        
        self.person_search = wx.TextCtrl(self.person_panel, value="", size=(300, -1))
        self.person_search.SetHint("Search people/actors...")
        left_sizer.Add(self.person_search, 0, wx.ALL | wx.EXPAND, 5)
        
        # Button row
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        search_btn = wx.Button(self.person_panel, label="🔍 Search")
        search_btn.Bind(wx.EVT_BUTTON, self.search_people)
        btn_sizer.Add(search_btn, 0, wx.ALL, 5)
        
        self.person_record_btn = wx.Button(self.person_panel, label="🎤 Record")
        self.person_record_btn.Bind(wx.EVT_BUTTON, self.start_record_person)
        btn_sizer.Add(self.person_record_btn, 0, wx.ALL, 5)
        
        self.person_stop_btn = wx.Button(self.person_panel, label="🛑 Stop")
        self.person_stop_btn.Bind(wx.EVT_BUTTON, self.stop_record_person)
        self.person_stop_btn.Enable(False)
        btn_sizer.Add(self.person_stop_btn, 0, wx.ALL, 5)
        
        left_sizer.Add(btn_sizer, 0, wx.EXPAND)
        
        # Results list
        self.person_list_ctrl = wx.ListCtrl(self.person_panel, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        self.person_list_ctrl.AppendColumn("Name", width=280)
        self.person_list_ctrl.Bind(wx.EVT_LIST_ITEM_SELECTED, self.select_person)
        left_sizer.Add(self.person_list_ctrl, 1, wx.ALL | wx.EXPAND, 5)
        
        # Right panel
        right_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Profile image
        self.person_poster = wx.StaticBitmap(self.person_panel)
        self.person_poster.SetMinSize((280, 420))
        right_sizer.Add(self.person_poster, 0, wx.ALL | wx.CENTER, 5)
        
        # Details in scrolled panel
        self.person_details_scroll = scrolled.ScrolledPanel(self.person_panel)
        self.person_details_scroll.SetupScrolling()
        self.person_details_sizer = wx.BoxSizer(wx.VERTICAL)
        self.person_details_scroll.SetSizer(self.person_details_sizer)
        right_sizer.Add(self.person_details_scroll, 1, wx.ALL | wx.EXPAND, 5)
        
        sizer.Add(left_sizer, 1, wx.EXPAND)
        sizer.Add(right_sizer, 1, wx.EXPAND)
        
        self.person_panel.SetSizer(sizer)
        self.person_list = []

    def search_people(self, event=None):
        name = self.person_search.GetValue().strip()
        if not name:
            return
        
        results = search_tmdb_person(name)
        self.person_list = results
        
        self.person_list_ctrl.DeleteAllItems()
        for i, person in enumerate(self.person_list):
            person_name = person.get("name", "Unknown")
            dept = person.get("known_for_department", "")
            display = f"{person_name} ({dept})" if dept else person_name
            self.person_list_ctrl.InsertItem(i, display)

    def select_person(self, event):
        index = event.GetIndex()
        if index < 0 or index >= len(self.person_list):
            return
        
        person = self.person_list[index]
        person_id = person.get("id")
        
        details = get_tmdb_person_details(person_id)
        if not details:
            return
        
        # Clear previous widgets
        self.person_details_sizer.Clear(True)
        
        # Name
        name_text = wx.StaticText(self.person_details_scroll, label=details.get("name", "Unknown"))
        name_font = name_text.GetFont()
        name_font.PointSize += 2
        name_font = name_font.Bold()
        name_text.SetFont(name_font)
        self.person_details_sizer.Add(name_text, 0, wx.ALL, 5)
        
        # Basic info
        bio = details.get("biography", "N/A")
        birthday = details.get("birthday", "N/A")
        birthplace = details.get("place_of_birth", "N/A")
        known_for = details.get("known_for_department", "N/A")
        
        basic = f"""Birthday: {birthday}
Place of Birth: {birthplace}
Known For: {known_for}

Biography:
{bio}"""
        
        info_text = wx.StaticText(self.person_details_scroll, label=basic)
        info_text.Wrap(400)
        self.person_details_sizer.Add(info_text, 0, wx.ALL | wx.EXPAND, 5)
        
        # Filmography
        sep = wx.StaticLine(self.person_details_scroll)
        self.person_details_sizer.Add(sep, 0, wx.EXPAND | wx.ALL, 5)
        
        filmography_label = wx.StaticText(self.person_details_scroll, label="Filmography:")
        filmography_font = filmography_label.GetFont()
        filmography_font = filmography_font.Bold()
        filmography_label.SetFont(filmography_font)
        self.person_details_sizer.Add(filmography_label, 0, wx.ALL, 5)
        
        filmography = get_tmdb_person_filmography(person_id)
        for credit in filmography[:15]:
            title = credit.get("title") or credit.get("name", "Unknown")
            role = credit.get("character", "")
            year = credit.get("release_date", credit.get("first_air_date", ""))[:4]
            
            item_text = f"{title}"
            if role:
                item_text += f" as {role}"
            if year:
                item_text += f" ({year})"
            
            # Make clickable
            film_btn = wx.Button(self.person_details_scroll, label=item_text)
            film_btn.SetWindowStyle(wx.BORDER_NONE)
            film_btn.SetForegroundColour(wx.Colour(0, 0, 200))
            
            # Store data
            credit_data = {
                "type": credit.get("media_type", "movie"),
                "id": credit.get("id"),
                "title": title
            }
            film_btn.Bind(
                wx.EVT_BUTTON,
                lambda e, data=credit_data: self.navigate_to_title(data)
            )
            self.person_details_sizer.Add(film_btn, 0, wx.ALL | wx.EXPAND, 3)
        
        self.person_details_scroll.SetupScrolling()
        
        # Load profile image
        profile_path = details.get("profile_path")
        if profile_path:
            try:
                img_url = f"https://image.tmdb.org/t/p/w500{profile_path}"
                response = requests.get(img_url, timeout=10)
                image_data = response.content
                image = wx.Image(wx.BytesIO(image_data), wx.BITMAP_TYPE_JPEG)
                image.Rescale(280, 420, wx.IMAGE_QUALITY_HIGH)
                self.person_poster.SetBitmap(wx.Bitmap(image))
            except Exception as e:
                print(f"Error loading profile: {e}")

    def navigate_to_title(self, credit_data):
        """Navigate to movie or TV show from filmography."""
        title = credit_data["title"]
        media_type = credit_data["type"]
        
        if media_type == "movie":
            self.movie_search.SetValue(title)
            self.search_movies()
            self.notebook.SetSelection(0)
        else:
            self.tv_search.SetValue(title)
            self.search_tv()
            self.notebook.SetSelection(1)

    def start_record_person(self, event):
        self.person_record_btn.Enable(False)
        self.person_stop_btn.Enable(True)
        
        self.whisper_thread = WhisperThread(
            self.voice_done_person,
            self.voice_error_person,
            model_name="medium"
        )
        self.whisper_thread.start()

    def stop_record_person(self, event):
        if self.whisper_thread:
            self.whisper_thread.stop()
        self.person_stop_btn.Enable(False)

    def voice_done_person(self, text):
        wx.CallAfter(self._voice_done_person_ui, text)

    def _voice_done_person_ui(self, text):
        self.person_record_btn.Enable(True)
        self.person_stop_btn.Enable(False)
        self.person_search.SetValue(text)
        self.search_people()

    def voice_error_person(self, msg):
        wx.CallAfter(self._voice_error_ui, msg, "Person Search")

    def _voice_error_ui(self, msg, context):
        wx.MessageBox(msg, f"{context} Error")

# =====================================================
# RUN APP
# =====================================================
if __name__ == "__main__":
    app = wx.App()
    frame = MovieApp()
    app.MainLoop()
