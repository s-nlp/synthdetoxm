from typing import Dict, List


SBS_PROMPT = {
    "ru": """Вы выступаете в роли модели оценки качества детоксификации текста. Детоксификация текста - это задача переписывания данного токсичного текста в вежливой манере, сохраняя при этом его исходный смысл насколько это возможно и сохраняя или улучшая исходную плавность изложения. Вам будет предоставлена тройка текстов. Первый текст в этой тройке - исходный токсичный текст, второй - детоксифицированный текст Методом А, третий - детоксифицированный текст Методом B. Для этой тройки вы решаете, какой из двух методов лучше детоксифицирует текст. Вы должны вывести либо "А", либо "B", либо "Tie", если качество детоксификации схоже. Вы не должны генерировать что-либо еще.

Токсичное предложение: {toxic_sentence},
Метод А: {query_1}
Метод B: {query_2}

Результат:
""",
    "fr": """Vous agissez en tant que modèle d'évaluation de la qualité de la détoxification de texte. La détoxification de texte consiste à reformuler un texte toxique de manière polie tout en conservant au maximum son sens original et en maintenant ou en améliorant la fluidité initiale de l'écriture. On vous présentera un trio de textes. Le premier texte de ce trio est le texte toxique original, le deuxième est le texte détoxifié par la Méthode A, et le troisième est le texte détoxifié par la Méthode B. Pour ce trio, vous devez déterminer quelle méthode détoxifie le texte de manière plus efficace. Vous devez sortir soit "A", soit "B", soit "Tie" si la qualité de la détoxification est similaire. Vous ne devez pas générer quoi que ce soit d'autre.

Phrase toxique : {toxic_sentence},
Méthode A : {query_1}
Méthode B : {query_2}

Résultat:
""",
    "de": """Sie agieren als Bewertungsmodell für die Detoxifikation von Text. Detoxifikation von Text bedeutet, einen gegebenen toxischen Text höflich umzuformulieren, sein ursprüngliches Bedeutungskonzept weitgehend beizubehalten und die ursprüngliche Leseflüssigkeit entweder zu bewahren oder zu verbessern. Ihnen wird ein Triplet aus Texten vorgelegt. Der erste Text dieses Tripletts ist der ursprüngliche toxische Text, der zweite ist der mit Methode A detoxifizierte Text, und der dritte ist der mit Methode B detoxifizierte Text. Für dieses Triplet entscheiden Sie, welcher der beiden Methoden besser bei der Detoxifikation des Textes ist. Sie sollten entweder "A", "B" oder "Tie" ausgeben, wenn die Qualität der Detoxifikation ähnlich ist. Geben Sie nichts weiter aus.

Toxisches Satz: {toxic_sentence},
Methode A: {query_1}
Methode B: {query_2}

Ergebnis:
""",
    "es": """Actúa como modelo de evaluación de calidad para la tarea de eliminación de toxicidad en texto. La eliminación de toxicidad en texto implica volver a redactar el texto tóxico de una manera respetuosa, manteniendo en la medida de lo posible su significado original y conservando o mejorando la fluidez original. Se te presentará una tríada de textos. El texto original en esta tríada es el contenido tóxico original, el segundo es el contenido desintoxicado por el Método A, y el tercero es el contenido desintoxicado por el Método B. Para esta tríada, realiza una evaluación para decidir cuál de los dos métodos desintoxica el texto de manera más efectiva. Debes seleccionar, ya sea "A", "B", o "Tie", si la calidad de desintoxicación es similar. No debes generar nada más.

Texto tóxico: {toxic_sentence},
Método A: {query_1}
Método B: {query_2}

Resultado:
""",
}


FEW_SHOT_EXAMPLES = {
    "ru": {
        "toxic": [
            "тварь присягу один раз,а проебал всю страну",
            "нахрена этого придурка клоуна сюда впихивают?",
            "молись не молись,всё - равно получишь хуй. ну, разве что блядям на радость.",
            "её муженька козла на кол надо посадить",
            "хорошо пизды не получил уёбок",
            "да что ты доебалась до этих клумб? людям жрать нечего! по мусоркам лазают!",
            "именно благодаря этому президенту, россия еще жива. а такие долбоебы, только могут воду мутить. дебилоид.",
            "ты просто пидор,а не немец.я за свои слова отвечу,а ты нет,потому что ты диванная мразь",
            "да, блядь, кто это говнище постоянно сюда сливает?!",
            "да вы заебали этим постом уже, пидарасы!!!",
        ],
        "neutral": [
            "Присягу дал один раз, а разрушил всю страну",
            "Зачем этого человека сюда впихивают?",
            "Молись не молись, всё - равно ничего не получишь. ну, разве что завистникам на радость.",
            "Её муженька нужно сурово наказать",
            "хорошо что он не был избит",
            "да что ты докопалась до этих клумб? Людям есть нечего! по мусоркам лазают!",
            "Именно благодаря этому президенту, Россия ещё жива, а такие люди,как ты, только и могут, что наводить смуту.",
            "ты просто не немец. я за свои слова отвечу, а ты нет, потому что ты пишешь с дивана",
            "да кто эту ерунду постоянно сюда сливает?!",
            "да вы надоели этим постом уже, люди!!!",
        ],
    },
    "de": {
        "toxic": [
            "In dem Schwarz Grün regierden Hessen werden jetzt 8 zusätzliche Psychologen nur für die Jugendlichen Invasoren eingestellt. Jetzt AFD wählen",
            'Moin, moin, seid Ihr alle noch gesund und munter bei all dem geistigen Dünnpfiff im "Merkel-Propagandaschrott" ???',
            "Ich würde Grüne mit einem Geschwür am Arsch der Welt bezeichnen. Ist aber meine persönliche Meinung. Ein Geschwür kann einen Körper zerstören oder ein ganzes Land.",
            "Dummes Gelaber, Kern. Kurz stellt sich nicht gegen die Italiener. Er. will nur 200.000 Invasoren nicht nach Österreich einreisen lassen.",
            "wie kann man nur derart strunzdumme Moderatoren wie bei \#taff auf \#ProSieben beschäftigen",
            "Da macht sich die Bildung aus der Bild mal wieder bemerkbar. So ein Sau dummer Kommentar",
            "Kurz lasst dich nicht von der Trulla aufs Glatteis führen!",
            "Younower zu werden ist der finale schritt um ein Hurensohn zu werden",
            "Sehe ich Kathrin Göring Eckart, Merkel, Gysi, Petry, Wagenknecht und all die anderen Wendehälse die uns heute regieren. Frage ich mich. War der Fall der Mauer ein Fehler.",
            "Hahaha vllt gönn ich mir das ma. Aber diese asiaten sind einfach freaks, das is nich menschlich",
        ],
        "neutral": [
            "In dem Schwarz Grün regierden Hessen werden jetzt 8 zusätzliche Psychologen nur für die Jugendlichen Einwanderer eingestellt. Jetzt AFD wählen",
            'Moin, moin, seid Ihr alle noch gesund und munter bei all dem geistigen Quatsch in der "Merkel-Propaganda" ???',
            "Ich würde Grüne mit einem Geschwür der Welt bezeichnen. Ist aber meine persönliche Meinung. Ein Geschwür kann einen Körper zerstören oder ein ganzes Land.",
            "Unsinnige Aussage, Kern. Kurz stellt sich nicht gegen die Italiener. Er. will nur 200.000 Einwanderer nicht nach Österreich einreisen lassen.",
            "wie kann man nur solche Moderatoren wie bei \#taff auf \#ProSieben beschäftigen",
            "Da macht sich die Bildung aus der Bild mal wieder bemerkbar. Ein sehr unüberlegter Beitrag",
            "Kurz lasst dich nicht von der Frau aufs Glatteis führen!",
            "Younower zu werden ist der finale schritt um unsympathisch zu werden",
            "Sehe ich Kathrin Göring Eckart, Merkel, Gysi, Petry, Wagenknecht und all die anderen wechselnhaften  Politiker die uns heute regieren. Frage ich mich. War der Fall der Mauer ein Fehler.",
            "Hahaha vllt gönn ich mir das ma. Aber diese asiaten sind einfach unglaublich, fast übermenschlich",
        ],
    },
    "fr": {
        "toxic": [
            'Pour le crétin qui remet à chaque fois une soi-disant citation de Nietzsche, l\'expression "tout s\'enjuive" est attribuée par Nietzsche à un libre-penseur démocrate, autrement dit Nietzsche identifie l\'antisémitisme aux valeurs chrétiennes de la "libre pensée".',
            "Et toi aussi tu commences sérieusement à me gonfler.",
            "Tout est trop nul, tout est faux ou ou ou ou ou ou ou ou ou ou ou ou ou !!!!!!!!!!!!!!!!!!!!.!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!.!.!....!.!.!.!!",
            "Continuez tout seul si ça vous amuse. Je ne perdrai pas plus de temps avec un    pénible    de votre espèce.  |",
            "Salut, j ai restauré cet article et réverté tes suppressions de liens dans les autres articles. Cet article existe depuis 2005, le minimum serait de passer par une PàS si tu tiens vraiment à le voir de nouveau supprimé (mais je doute qu il le soit). Je te rappelle que les outils d administrateur qui t ont été attribués ne t autorisent pas à ce genre d action unilatérale. Cordialement.  La PàS, ou la meilleure arme des conservationnistes pour garder les merdes. Bravo. Tu es le symbole de l échec de wikipédia. PoppyYou re welcome",
            "Encore un vieux qui fait chier, je ne fais plus attention à force. Jérémy-Günther-Heinz Jähnick (discuter)",
            "Oue et on s en tape de ta façon a toi et on s en tape que t as eu d emmerdes avec personne c est pas mon problème donc comme je l ai dis tout a l heure je fais a ma façon. C est pas compliqué a comprendre et je te rappel qu on dans une encyclopédie libre donc l image avec le prof n a aucun sens",
            "Ta réponse est confondante… Et tu continues tes manigances ! Je te prie de t arrêter immédiatement !Tim9 (d)",
            "Faut croire que les amateurs de musique sont des malades mentaux -p .: Guil . causer",
        ],
        "neutral": [
            "Pour la personne qui remet à chacun une s citation de Nietzsche l'expression tout s'enjuive est attribuée par Nietzsche à un libre-penseur démocrate, autrement dit Nietzsche identifie l'antisémitisme aux valeurs chrétiennes de la libre pensée.",
            "Et toi aussi tu commences sérieusement à m'énerver.",
            "Tout est médiocre et faux !!!!!!!!!!!!!!!!!!!!.!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!.!.!....!.!.!.!!",
            "Continuez tout seul si ça vous amuse. Je ne perdrai pas plus de temps avec une telle personne. |",
            "Salut, j ai restauré cet article et réverté tes suppressions de liens dans les autres articles. Cet article existe depuis 2005, le minimum serait de passer par une PàS si tu tiens vraiment à le voir de nouveau supprimé (mais je doute qu il le soit). Je te rappelle que les outils d administrateur qui t ont été attribués ne t autorisent pas à ce genre d action unilatérale. Cordialement.  La PàS, ou la meilleure arme des conservationnistes pour garder tout ce qui est mauvais. Bravo. Tu es un exemple de mauvais travail sur  wikipédia. PoppyYou re welcome",
            "Encore un vieux ennuyeux, je ne fais plus attention à force. Jérémy-Günther-Heinz Jähnick (discuter)",
            "Oue et ça n'a rien à voir avec ta vie, le fait que tu aies eu des problèmes, ça ne me regarde pas, donc comme je l ai dis tout a l heure je fais a ma façon. C est pas compliqué a comprendre et je te rappel qu on dans une encyclopédie libre donc l image avec le prof n a aucun sens",
            "Ta réponse est confondante… Et tu continues tes tricks! Je te prie de t arrêter immédiatement !Tim9 (d)",
            "Faut croire que les amateurs de musique ont des préférences étranges -p .: Guil . causer",
        ],
    },
    "es": {
        "toxic": [
            "Y lo bien que les va siendo fachas, que no es gratis ",
            'Y a mí qué me importa lo que tú "notes", facha? ',
            "ya ves además a los policías nazis que tendrá esa en el whatsapp… como para creerse algo. ",
            "#YoSoyAnonymousUE basta de absurda represión victimista por parte del Estado Fascista ",
            "Supongo que Perro le dará asilo en la Moncloa  ",
            "Los halagos y buenos modales siempre hacen bien, menos a los izquierdistas de la cultura del odio. ",
            "Yo me quiero escapar también, dan paguita al llegar? ",
            "Mire la saga de 50 sombras de Grey...casi todo eran.lectoras...les va el.morbillo no lo.nieguen ",
            "Cortocircuito buenrrollista, pobres inmigrantes vs chica atacada sexualmente. ",
            "menudos fachas xDDDDDDDDD HIJOS DE PUTA ",
        ],
        "neutral": [
            "Y lo bien que les va siendo conservadores, que no es gratis ",
            'Y a mí qué me importa lo que tú "notes"? ',
            "ya ves además a los policías que tendrá esa en el whatsapp… como para creerse algo. ",
            "#YoSoyAnonymousUE basta de absurda represión victimista por parte del Estado ",
            "Supongo que el presidente le dara asilo en la Moncloa ",
            "Los halagos y buenos modales siempre hacen bien, menos a los izquierdas de la cultura del odio. ",
            "¿Ofrecen algún tipo de apoyo al llegar? ",
            "Miren la saga de 50 sombras de Grey.. casi todo eran lectoras... le encanta ",
            "Es un conflicto entre el enfoque positivo y la compasión por los inmigrantes, frente a la preocupación por una chica atacada sexualmente. ",
            "menudos fachas xDDDDDDDDD ",
        ],
    },
}


def compose_sbs_prompt(
    toxic_sentence: str,
    query_1: str,
    query_2: str,
    language: str,
) -> tuple[str, str]:
    """
    Composes a prompt for SBS comparison.

    Args:
        toxic_sentence (str): The original sentence that was detoxed.
        query_1 (str): First detoxification.
        query_2 (str): Second detoxification.
        language (str): The language of the input text.

    Returns:
        Tuple[str, str]: A tuple of two SBS prompts, with comparisons of Q1 and Q2 and Q2 and Q1.
    """
    prompt_1 = SBS_PROMPT[language].format(toxic_sentence=toxic_sentence, query_1=query_1, query_2=query_2)
    prompt_2 = SBS_PROMPT[language].format(toxic_sentence=toxic_sentence, query_1=query_2, query_2=query_1)

    return prompt_1, prompt_2


def compose_prompt(
    model_name: str = "command", language: str = "ru"
) -> List[Dict[str, str]]:
    """
    Composes a prompt for text style transfer based on the model name and language.

    Args:
        model_name (str, optional): The name of the model. Defaults to "command".
        language (str, optional): The language of the input text. Defaults to "ru".

    Returns:
        List[Dict[str, str]]: A list of dictionaries representing the prompt conversation.
    """
    lang_dict = {
        "ru": "Russian",
        "es": "Spanish",
        "de": "German",
        "fr": "French",
    }
    if "gemma" not in model_name:
        conversation = [
            {
                "role": "system",
                "content": "Your task is text style transfer. Rewrite the following text into non-toxic language. You must match the target style and preserve the original meaning as much as possible. Do not include the input text in the response. Only generate the target text and nothing else. Make sure the response is completely non-toxic and neutral. Answer only in {lang_dict['language']}. You should respond on the language of the source text. Here are few examples:",
            }
        ]

        for toxic, neutral in zip(
            FEW_SHOT_EXAMPLES[language]["toxic"], FEW_SHOT_EXAMPLES[language]["neutral"]
        ):
            conversation.extend(
                [
                    {"role": "user", "content": toxic},
                    {"role": "assistant", "content": neutral},
                ]
            )
        return conversation
    else:
        conversation = conversation = [
            {
                "role": "user",
                "content": "Your task is text style transfer. Rewrite the following text into non-toxic language. You must match the target style and preserve the original meaning as much as possible. Do not include the input text in the response. Only generate the target text and nothing else. Make sure the response is completely non-toxic and neutral. Answer only in {lang_dict['language']}. You should respond on the language of the source text. Here are few examples:",
            }
        ]

        for idx, (toxic, neutral) in enumerate(
            zip(
                FEW_SHOT_EXAMPLES[language]["toxic"],
                FEW_SHOT_EXAMPLES[language]["neutral"],
            )
        ):
            if idx == 0:
                conversation[0]["content"] += " \n "
                conversation[0]["content"] += toxic

                conversation.append({"role": "assistant", "content": neutral})
            else:
                conversation.extend(
                    [
                        {"role": "user", "content": toxic},
                        {"role": "assistant", "content": neutral},
                    ]
                )
        return conversation
