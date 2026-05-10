type_pixel = 'type_pixel'
type_bar = 'type_bar'
type_blur = 'type_blur'
type_outline = 'type_outline'

legacy_nudenet_v3_classes = [
        'VULVA_COVERED',
        'FACE_FEMME',
        'BUTTOCKS_EXPOSED',
        'FEMME_BREAST_EXPOSED',
        'VULVA_EXPOSED',
        'MASC_BREAST_EXPOSED',
        'ANUS_EXPOSED',
        'FEET_EXPOSED',
        'BELLY_COVERED',
        'FEET_COVERED',
        'ARMPIT_COVERED',
        'ARMPIT_EXPOSED',
        'FACE_MASC',
        'BELLY_EXPOSED',
        'PENIS_EXPOSED',
        'ANUS_COVERED',
        'BREAST_COVERED',
        'BUTTOCKS_COVERED',
    ]

# Keep the historic name for older code paths that still import it directly.
nudenet_v3_classes = legacy_nudenet_v3_classes

model_profile_auto   = 'auto'
model_profile_small  = 'small'
model_profile_medium = 'medium'
model_profile_large  = 'large'

model_class_aliases = {
        'FEMALE_GENITALIA_COVERED': 'VULVA_COVERED',
        'FACE_FEMALE': 'FACE_FEMME',
        'FEMALE_BREAST_EXPOSED': 'FEMME_BREAST_EXPOSED',
        'FEMALE_GENITALIA_EXPOSED': 'VULVA_EXPOSED',
        'MALE_BREAST_EXPOSED': 'MASC_BREAST_EXPOSED',
        'ARMPITS_COVERED': 'ARMPIT_COVERED',
        'ARMPITS_EXPOSED': 'ARMPIT_EXPOSED',
        'FACE_MALE': 'FACE_MASC',
        'MALE_GENITALIA_EXPOSED': 'PENIS_EXPOSED',
        'FEMALE_BREAST_COVERED': 'BREAST_COVERED',
    }

model_profiles = {
        model_profile_small: {
            'label': 'Small',
            'basename': '320n',
            'description': 'Fastest profile for CPU, DirectML, and low-VRAM systems.',
            'default-net-sizes': [ 640 ],
            'fallback_profiles': [ model_profile_medium ],
            'classes': [
                'FEMALE_GENITALIA_COVERED',
                'FACE_FEMALE',
                'BUTTOCKS_EXPOSED',
                'FEMALE_BREAST_EXPOSED',
                'FEMALE_GENITALIA_EXPOSED',
                'MALE_BREAST_EXPOSED',
                'ANUS_EXPOSED',
                'FEET_EXPOSED',
                'BELLY_COVERED',
                'FEET_COVERED',
                'ARMPITS_COVERED',
                'ARMPITS_EXPOSED',
                'FACE_MALE',
                'BELLY_EXPOSED',
                'MALE_GENITALIA_EXPOSED',
                'ANUS_COVERED',
                'FEMALE_BREAST_COVERED',
                'BUTTOCKS_COVERED',
            ],
        },
        model_profile_medium: {
            'label': 'Medium',
            'basename': '640m',
            'description': 'Best default balance of quality, VRAM use, and latency.',
            'default-net-sizes': [ 1280, 640 ],
            'fallback_profiles': [ model_profile_small ],
            'classes': [
                'FEMALE_GENITALIA_COVERED',
                'FACE_FEMALE',
                'BUTTOCKS_EXPOSED',
                'FEMALE_BREAST_EXPOSED',
                'FEMALE_GENITALIA_EXPOSED',
                'MALE_BREAST_EXPOSED',
                'ANUS_EXPOSED',
                'FEET_EXPOSED',
                'BELLY_COVERED',
                'FEET_COVERED',
                'ARMPITS_COVERED',
                'ARMPITS_EXPOSED',
                'FACE_MALE',
                'BELLY_EXPOSED',
                'MALE_GENITALIA_EXPOSED',
                'ANUS_COVERED',
                'FEMALE_BREAST_COVERED',
                'BUTTOCKS_COVERED',
            ],
        },
        model_profile_large: {
            'label': 'Large',
            # NudeNet does not currently ship a larger official PyTorch weight than 640m,
            # so the large profile reuses 640m and relies on more aggressive sizing.
            'basename': '640m',
            'description': 'Highest-quality preset; uses the 640m family with more aggressive sizing.',
            'default-net-sizes': [ 1280, 640, 2560 ],
            'fallback_profiles': [ model_profile_medium, model_profile_small ],
            'classes': [
                'FEMALE_GENITALIA_COVERED',
                'FACE_FEMALE',
                'BUTTOCKS_EXPOSED',
                'FEMALE_BREAST_EXPOSED',
                'FEMALE_GENITALIA_EXPOSED',
                'MALE_BREAST_EXPOSED',
                'ANUS_EXPOSED',
                'FEET_EXPOSED',
                'BELLY_COVERED',
                'FEET_COVERED',
                'ARMPITS_COVERED',
                'ARMPITS_EXPOSED',
                'FACE_MALE',
                'BELLY_EXPOSED',
                'MALE_GENITALIA_EXPOSED',
                'ANUS_COVERED',
                'FEMALE_BREAST_COVERED',
                'BUTTOCKS_COVERED',
            ],
        },
    }

supported_model_profiles = [ model_profile_auto ] + list( model_profiles.keys() )

single_pass = 'single_pass'
no_overlap  = 'no_overlap'

supported_sizes = [ 640, 1280, 1920, 2560 ]

def normalize_model_profile( profile, default=model_profile_medium ):
    default = str(default).strip().lower() if default is not None else model_profile_medium
    if default not in supported_model_profiles:
        default = model_profile_medium
    normalized = str(profile).strip().lower() if profile is not None else ''
    if normalized in supported_model_profiles:
        return normalized
    return default

def normalize_detection_class( class_name ):
    if class_name is None:
        return None
    normalized = str(class_name).strip()
    return model_class_aliases.get( normalized, normalized )

def _iter_model_classes( classes ):
    if isinstance( classes, dict ):
        for key in sorted( classes ):
            yield key, classes[ key ]
        return

    for i, class_name in enumerate( classes ):
        yield i, class_name

def get_detection_classes():
    classes = list( legacy_nudenet_v3_classes )
    for profile in model_profiles:
        for class_name in model_profiles[ profile ].get( 'classes', [] ):
            normalized = normalize_detection_class( class_name )
            if normalized not in classes:
                classes.append( normalized )
    return classes

def get_detection_class_index_map( model_classes, known_classes=None ):
    if known_classes is None:
        known_classes = get_detection_classes()
    known_index = { class_name: i for i, class_name in enumerate( known_classes ) }
    out = {}
    for raw_index, class_name in _iter_model_classes( model_classes ):
        normalized = normalize_detection_class( class_name )
        out[ raw_index ] = known_index.get( normalized )
    return out

def get_model_profile_base_names():
    base_names = []
    for profile in model_profiles:
        basename = model_profiles[ profile ][ 'basename' ]
        if basename not in base_names:
            base_names.append( basename )
    return base_names
