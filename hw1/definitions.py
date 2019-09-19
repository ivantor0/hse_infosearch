import os
import re
import pymystem3
import pandas as pd

from collections import OrderedDict, Counter
from collections.abc import Iterable
from random import choice
from numpy import nan as NaN


class Indexer:
    
    def __init__(self, start_path):
        print("Построение индекса… (занимает около минуты)")
        self._m = pymystem3.mystem.Mystem()
        self.build_index(start_path)
    
    def build_index(self, start_path):
        adhoc_dict = {"season": [], "serie": [], "title": [], "counter": []}
        for root, dirs, files in os.walk("friends"):
            for file in files:
                fstruct = re.search(r"^Friends - ((.*?)x.*?) - (.*?)\.ru\.txt$", file)
                adhoc_dict["season"].append(int(fstruct.group(2)))
                adhoc_dict["serie"].append(fstruct.group(1))
                adhoc_dict["title"].append(fstruct.group(3))
                with open(os.path.join(root, file), "r", encoding="utf-8-sig") as f:
                    adhoc_dict["counter"].append(self.count_file(f.read()))
        
        # general structure
        self.general_index = adhoc_dict["counter"][0].copy()
        for i in range(1, len(adhoc_dict["counter"])):
            self.general_index += adhoc_dict["counter"][i]
        
        # series structure
        Words = sorted(list(self.general_index.keys()))
        series_dict = OrderedDict()
        for word in Words:
            series_dict[word] = []
        self.idx_to_series = {i: adhoc_dict["serie"][i] for i in range(len(adhoc_dict["serie"]))}
        self.series_to_idx = {adhoc_dict["serie"][i]: i for i in range(len(adhoc_dict["serie"]))}
        for c in adhoc_dict["counter"]:
            for word in Words:
                series_dict[word].append(c[word])
        self.series_index = pd.DataFrame(series_dict)
        
        # seasons structure
        seasons = sorted(list(set(adhoc_dict["season"])))
        series_to_seasons = {}
        for i in range(len(adhoc_dict["season"])):
            if adhoc_dict["season"][i] not in series_to_seasons:
                series_to_seasons[adhoc_dict["season"][i]] = []
            series_to_seasons[adhoc_dict["season"][i]].append(adhoc_dict["serie"][i])
        adhoc_ordereddict = OrderedDict()
        for word in Words:
            adhoc_ordereddict[word] = []
        self.seasons_index = pd.DataFrame(adhoc_ordereddict)
        for season in series_to_seasons:
            begins_here = True
            for serie in series_to_seasons[season]:
                if begins_here:
                    season_array = self.series_index.loc[self.series_to_idx[serie]]
                    begins_here = False
                else:
                    season_array += self.series_index.loc[self.series_to_idx[serie]]
            self.seasons_index.loc[len(self.seasons_index)] = season_array
    
    def count_file(self, text_str):
        wordsoup = [word for word in self._m.lemmatize(text_str) if re.search(r"\w", word)]
        return Counter(wordsoup)
    
    def query(self, query_iterable, level="general"):
        if level not in ("general", "seasons", "series"):
            raise ValueError("This class can only search by series, by seasons or in general total.")
        if isinstance(query_iterable, str):
            query = [query_iterable.lower()]
        elif not isinstance(query_iterable, Iterable):
            raise TypeError("Query must be either a string or any iterable from strings.")
        else:
            query = [q.lower() for q in query_iterable]
        if level == "general":
            return [self.general_index[elem] for elem in query]
        elif level == "seasons":
            return [[(i+1, int(self.seasons_index[elem][i])) for i in range(len(self.seasons_index[elem]))] for elem in query]
        else:
            return [[(self.idx_to_series[i], int(self.series_index[elem][i])) for i in range(len(self.series_index[elem]))] for elem in query]


class Solver:
    
    def __init__(self, indexer):
        self.index = indexer
    
    def taskA(self):
        #  a) какое слово является самым частотным
        query = self.index.general_index.most_common(1)
        print("a) какое слово является самым частотным?")
        print("'"+query[0][0]+"'", "с частотностью", query[0][1])
    
    def taskB(self):
        #  b) какое самым редким
        query = self.index.general_index.most_common(len(self.index.general_index))
        query.reverse()
        rarest_words = [query[0][0]]
        least_occurrences = query[0][1]
        for i in range(1, len(query)):
            if query[i][1] > least_occurrences:
                break
            rarest_words.append(query[i][0])
        words_to_select = 10
        found = set()
        for _ in range(words_to_select):
            new_word = choice(rarest_words)
            while new_word in found:
                new_word = choice(rarest_words)
            found.add(new_word)
        print("b) какое слово является самым редким?")
        print("Выбрано случайных слов:", words_to_select)
        for word in found:
            print("'"+word+"'", "с частотностью", least_occurrences)
    
    def taskC(self):
        #  c) какой набор слов есть во всех документах коллекции
        nan_df = self.index.series_index.replace(0, NaN)
        wordstring = ", ".join(list(nan_df.dropna(axis=1).columns))
        print("c) какой набор слов есть во всех документах коллекции?")
        print(wordstring)
    
    def taskD(self):
        #  d) какой сезон был самым популярным у Чендлера? у Моники?
        characters = ["Чендлер", "Моника"]
        query = self.index.query(characters, level="seasons")
        print("d) какой сезон был самым популярным у Чендлера? у Моники?")
        for i in range(len(characters)):
            character = characters[i]
            max_hits = -1
            best_season = -1
            for season, hits in query[i]:
                if hits > max_hits:
                    best_season = season
                    max_hits = hits
            print(character, ": лучший сезон - ", best_season, " (", max_hits, ")",  sep="")
    
    def taskE(self):
        #  e) кто из главных героев статистически самый популярный?
        characters = ["Чендлер", "Моника", "Рейчел", "Росс", "Фиби", "Джоуи"]
        most_popular_character = None
        max_hits = -1
        for result in zip(characters, self.index.query(characters)):
            character = result[0]
            hits = result[1]
            if hits > max_hits:
                max_hits = hits
        print("e) кто из главных героев статистически самый популярный?")
        print("Чаще всех остальных упомиинается", character, "с частотностью", max_hits)
        
    def solve(self):
        print("Индекс построен, выполняем решение заданий")
        print()
        for i in range(ord("A"), ord("F")):
            eval("self.task"+chr(i)+"()")
            print()
