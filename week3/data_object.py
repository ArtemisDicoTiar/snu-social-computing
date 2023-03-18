from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class Url:
    url: str
    expanded_url: str
    display_url: str
    indices: List[int]


@dataclass
class TweetEntity:
    hashtags: List[str]
    symbols: List[str]
    user_metions: List[str]
    urls: List[Url]


@dataclass
class Metadata:
    result_type: str
    iso_language_code: str


@dataclass
class UserEntity:
    description: Dict[str, list]


@dataclass
class User:
    id: int
    id_str: str
    name: str
    screen_name: str
    location: str
    description: str
    url: Optional[str]
    entities: UserEntity
    protected: bool
    followers_count: int
    friends_count: int
    listed_count: int
    created_at: str
    favourites_count: int
    utc_offset: Optional[int]
    time_zone: Optional[int]
    geo_enabled: bool
    verified: bool
    statuses_count: int
    lang: Optional[str]
    contributors_enabled: bool
    is_translator: bool
    is_translation_enabled: bool
    profile_background_color: str
    profile_background_image_url: str
    profile_background_image_url_https: str
    profile_background_tile: str
    profile_image_url: str
    profile_image_url_https: str
    profile_link_color: str
    profile_sidebar_border_color: str
    profile_sidebar_fill_color: str
    profile_text_color: str
    profile_use_background_image: bool
    has_extended_profile: bool
    default_profile: bool
    default_profile_image: bool
    following: bool
    follow_request_sent: bool
    notifications: bool
    translator_type: str
    withheld_in_countries: list


@dataclass
class Tweet:
    # when did the tweet created
    created_at: str

    # id of the tweet
    id: int
    id_str: str

    # content of the tweet
    text: str
    truncated: bool

    entities: TweetEntity

    metadata: Metadata

    source: str

    in_reply_to_status_id: Optional[int]
    in_reply_to_status_id_str: Optional[str]
    in_reply_to_user_id: Optional[int]
    in_reply_to_user_id_str: Optional[str]
    in_reply_to_screen_name: Optional[str]

    user: User
    geo: Optional[str]
    coordinates: Optional[str]
    place: Optional[str]
    contributors: Optional[str]
    is_quote_status: bool
    retweet_count: int
    favorite_count: int
    favorited: bool
    retweeted: bool
    possibly_sensitive: bool
    lang: str

