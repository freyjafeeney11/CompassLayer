TRANSLATIONS = {
    'fr': {
        'ping_rate': 'Vitesse du ping',
        'ping_volume': 'Volume du ping',
        'tts_volume': 'Volume vocal',
        'tts_speed': 'Vitesse vocale',
        'language': 'Langue',
        'lang_value_0': 'Français',
        'lang_value_1': 'Anglais',
        'unit_times': 'fois',
        'unit_none': '',
        'menu_opened': 'Menu paramètres. Utilisez les flèches haut et bas pour choisir, et gauche et droite pour ajuster.',
        'menu_closed': 'Menu fermé.',
        'help_controls': 'Commandes: F 6 pour entendre ce message. F 7 pour les paramètres. Majuscule F 8 pour balayer. Majuscule F 9 pour quitter le programme.',
        'arrived': "Vous êtes arrivé à l'objectif.",
        'scan_mode': 'Balayage en cours',
        'main_quest': 'Objectif principal',
        'treasure': 'Trésor',
        'stockpile': 'Réserve',
        'straight': 'droit devant',
        'left': 'gauche',
        'right': 'droite',
        'far_to': 'loin à {side}',
        'to_side': 'à {side}',
        'slightly_to': 'légèrement à {side}',
        'meters': 'à {dist} mètre{plural}',
        'contains_resources': ', contient du bois, des cultures ou des minéraux',
        'startup_msg': 'Lancement de Compass Layer. Appuyez sur F6 pour entendre les commandes.'
    },
    'en': {
        'ping_rate': 'Ping rate',
        'ping_volume': 'Ping volume',
        'tts_volume': 'Voice volume',
        'tts_speed': 'Voice speed',
        'language': 'Language',
        'lang_value_0': 'French',
        'lang_value_1': 'English',
        'unit_times': 'times',
        'unit_none': '',
        'menu_opened': 'Settings menu. Use up and down arrows to choose, and left and right to adjust.',
        'menu_closed': 'Menu closed.',
        'help_controls': 'Controls: F 6 to hear this message. F 7 for settings. Shift F 8 to sweep. Shift F 9 to exit the program.',
        'arrived': "You have arrived at the objective.",
        'scan_mode': 'Scanning',
        'main_quest': 'Main objective',
        'treasure': 'Treasure',
        'stockpile': 'Stockpile',
        'straight': 'straight ahead',
        'left': 'left',
        'right': 'right',
        'far_to': 'far to the {side}',
        'to_side': 'to the {side}',
        'slightly_to': 'slightly to the {side}',
        'meters': 'at {dist} meter{plural}',
        'contains_resources': ', contains wood, crops, or minerals',
        'startup_msg': 'Compass Layer launched. Press F6 to hear controls.'
    }
}

_current_lang = 'fr'

def set_lang(lang: str):
    global _current_lang
    _current_lang = lang if lang in TRANSLATIONS else 'fr'

def get_lang() -> str:
    return _current_lang

def get_text(key: str, **kwargs) -> str:
    text = TRANSLATIONS[_current_lang].get(key, key)
    if kwargs:
        return text.format(**kwargs)
    return text
