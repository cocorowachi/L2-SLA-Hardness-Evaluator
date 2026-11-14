# l2diff initial evaluation

Artifacts: `data/embed_all`. Task A is reported on the full vectors (`data/embed_all`); the compact release (`artifacts/release`) reproduces Tasks B and C exactly but has a smaller vocabulary, so Task A coverage and P@k differ on it and its raw-vector baseline is not included.

## Task A: translation retrieval (MUSE test dictionaries)

| pair | n | P@1 | P@5 | P@10 | raw P@1 | raw P@10 |
|---|---|---|---|---|---|---|
| en-es | 1072 | 0.682 | 0.798 | 0.84 | 0.672 | 0.835 |
| es-en | 994 | 0.688 | 0.806 | 0.842 | 0.652 | 0.794 |
| en-ja | 898 | 0.193 | 0.323 | 0.374 | 0.16 | 0.333 |
| ja-en | 871 | 0.215 | 0.324 | 0.371 | 0.177 | 0.309 |

## Task B: difficulty detection (hard vs easy cases)

| set | n | hard | AUC difficulty | AUC gap only | AUC split only | AUC baseline | acc | baseline acc |
|---|---|---|---|---|---|---|---|---|
| test | 40 | 19 | 0.594 [0.416, 0.779] | 0.897 [0.779, 0.99] | 0.604 [0.421, 0.789] | 0.982 [0.94, 1.0] | 0.65 | 0.925 |
| dev | 15 | 12 | 0.889 [0.583, 1.0] | 0.861 [0.639, 1.0] | 0.778 [0.472, 1.0] | 0.944 [0.778, 1.0] | 0.8 | 0.733 |
| all | 55 | 31 | 0.688 [0.535, 0.821] | 0.882 [0.778, 0.962] | 0.656 [0.496, 0.804] | 0.948 [0.879, 0.993] | 0.691 | 0.873 |

Missing from vocabulary: none

### Per case

| id | split | dir | lemma | label | type | gap | split | difficulty | baseline | nearest L2 word | rebuilt from |
|---|---|---|---|---|---|---|---|---|---|---|---|
| en-television-ja | test | ja->en | television | easy | loanword | 34.1 | 98.2 | 98.2 | 3.4 | TV | テレビ, 放送, 漫画 |
| ja-tv | test | en->ja | テレビ | easy | loanword | 24.4 | 87.4 | 87.4 | 3.1 | 放送 | television, TV, newspaper |
| en-hotel-ja | test | ja->en | coffee | easy | loanword | 19.7 | 84.7 | 84.7 | 10.0 | chocolate | コーヒー, 携帯, 空気 |
| es-cafe-ja | test | ja->es | café | easy | loanword | 4.0 | 81.7 | 81.7 | 12.5 | chocolate | コーヒー, カフェ, パン |
| en-president | test | es->en | president | easy | cognate | 5.7 | 79.5 | 79.5 | 15.1 | chairman | presidente, presidenta, rector |
| es-importante | test | en->es | importante | easy | cognate | 12.9 | 73.0 | 73.0 | 4.7 | relevante | important, significant, large |
| en-family | test | es->en | family | easy | cognate | 4.1 | 67.8 | 67.8 | 8.4 | household | familia, familiar, matrimonio |
| es-musica | test | en->es | música | easy | cognate | 30.5 | 66.4 | 66.4 | 24.8 | musical | music, musical, song |
| en-music | test | es->en | music | easy | cognate | 66.2 | 11.1 | 66.2 | 23.0 | musician | música, musical, musicología |
| ja-game | test | en->ja | ゲーム | easy | loanword | 2.3 | 51.0 | 51.0 | 15.6 | コンテンツ | game, gaming, Computer |
| ja-camera-es | test | es->ja | カメラ | easy | loanword | 22.8 | 48.8 | 48.8 | 24.0 | 写真 | cámara, telescopio, fotógrafo |
| en-team-ja | test | ja->en | team | easy | loanword | 7.9 | 41.9 | 41.9 | 1.4 | crew | チーム, 編隊, 戦力 |
| ja-team | test | en->ja | チーム | easy | loanword | 1.7 | 40.1 | 40.1 | 1.4 | グループ | team, group, club |
| en-game-ja | test | ja->en | game | easy | loanword | 22.0 | 38.4 | 38.4 | 16.6 | match | ゲーム, ボール, イベント |
| ja-pan-es | test | es->ja | パン | easy | loanword | 29.2 | 5.5 | 29.2 | 63.1 | パンチ | pasta, coca, café |
| es-hotel-ja | test | ja->es | hotel | easy | loanword | 19.2 | 25.0 | 25.0 | 1.3 | hospital | ホテル, 家屋, 企業 |
| es-universidad | test | en->es | universidad | easy | cognate | 20.3 | 24.2 | 24.2 | 41.5 | academia | university, United, municipality |
| ja-sport | test | en->ja | スポーツ | easy | loanword | 0.8 | 21.9 | 21.9 | 1.6 | サッカー | sport, Industrial, entertainment |
| en-nation | test | es->en | nation | easy | cognate | 18.2 | 17.1 | 18.2 | 18.2 | country | nación, país, nacional |
| es-animal | test | en->es | animal | easy | cognate | 2.9 | 10.6 | 10.6 | 3.9 | alimento | animal, fish, clothes |
| es-hospital | test | en->es | hospital | easy | cognate | 4.8 | 4.4 | 4.8 | 2.1 | hotel | hospital, institute |
| en-would | test | ja->en | would | hard | gap | 81.7 | 92.7 | 92.7 | 95.1 | will | です, 有る, 予想 |
| en-will | test | es->en | will | hard | gap | 83.3 | 85.2 | 85.2 | 87.7 | would | ir, poder, soler |
| en-have-aux | test | ja->en | have | hard | gap | 38.2 | 84.1 | 84.1 | 81.3 | be | 居る, 近年, 有る |
| en-do | test | es->en | do | hard | gap | 28.0 | 77.0 | 77.0 | 95.1 | will | no, hacer, tampoco |
| ja-kureru-es | test | es->ja | 呉れる | hard | split | 57.4 | 74.7 | 74.7 | 58.9 | 貰う | regalar, brindar, ayudar |
| es-clitic-ja | test | ja->es | él | hard | gap | 38.4 | 74.3 | 74.3 | 82.7 | el | を, の, れる |
| en-at | test | ja->en | at | hard | split | 68.2 | 54.1 | 68.2 | 97.8 | upon | に, で, 受け取る |
| ja-hiki-es | test | es->ja | 匹 | hard | gap | 34.2 | 68.0 | 68.0 | 87.0 | 隻 | especie, hectárea, kg |
| ja-kureru | test | en->ja | 呉れる | hard | split | 62.4 | 52.3 | 62.4 | 76.1 | 貰う | provide, friendly, dare |
| es-por-ja | test | ja->es | por | hard | split | 53.5 | 62.1 | 62.1 | 93.4 | porque | 因る, で, 何故 |
| en-any | test | ja->en | any | hard | gap | 54.4 | 60.8 | 60.8 | 60.6 | anything | 如何なる, 誰, あらゆる |
| en-tell | test | es->en | tell | hard | split | 58.2 | 47.6 | 58.2 | 62.8 | teach | decir, informar, narrar |
| es-clitic | test | en->es | él | hard | split | 51.5 | 54.9 | 54.9 | 92.5 | el | be, the, one |
| ja-hiki | test | en->ja | 匹 | hard | gap | 36.7 | 52.5 | 52.5 | 92.1 | 隻 | pound, breed, hectare |
| ja-ageru | test | en->ja | 上げる | hard | split | 46.6 | 51.6 | 51.6 | 67.5 | 取り上げる | cite, add, give |
| ja-mai | test | en->ja | 枚 | hard | gap | 38.2 | 38.6 | 38.6 | 91.4 | 冊 | litre, pair, piece |
| en-make | test | es->en | make | hard | split | 34.4 | 27.3 | 34.4 | 29.4 | making | hacer, hecho, crear |
| es-por | test | en->es | por | hard | split | 11.3 | 10.0 | 11.3 | 58.0 | porque | by, for, porous |
| es-para | test | en->es | para | hard | split | 9.7 | 0.7 | 9.7 | 48.5 | parar | for, to, parade |
| dev-energia | dev | en->es | energía | easy | cognate | 19.6 | 73.4 | 73.4 | 14.9 | electricidad | energy, electricity, hydroelectricity |
| dev-water | dev | ja->en | water | easy | translation | 27.6 | 25.8 | 27.6 | 6.4 | liquid | 水, 空気, 塩 |
| dev-hotel | dev | en->ja | ホテル | easy | loanword | 0.2 | 23.5 | 23.5 | 0.7 | レストラン | hotel, hostel, airport |
| es-la-ja | dev | ja->es | el | hard | gap | 34.2 | 99.8 | 99.8 | 12.9 | de | の, トリプル, チャイルド |
| dev-ser | dev | en->es | ser | hard | split | 5.0 | 96.0 | 96.0 | 9.8 | estar | be, become, correspond |
| dev-conocer | dev | en->es | conocer | hard | split | 45.3 | 91.5 | 91.5 | 20.2 | conocido | know, nickname, describe |
| dev-on | dev | ja->en | on | hard | split | 89.9 | 18.4 | 89.9 | 93.2 | upon | 初日, に, 上 |
| dev-ga | dev | en->ja | が | hard | split | 49.1 | 88.8 | 88.8 | 56.7 | を | be, of, about |
| dev-wo | dev | en->ja | を | hard | gap | 29.7 | 88.8 | 88.8 | 82.9 | が | a, of, successfully |
| dev-shu | dev | en->ja | 種 | hard | gap | 84.7 | 57.2 | 84.7 | 62.7 | 亜種 | breed, species, type |
| dev-wa | dev | en->ja | は | hard | split | 47.4 | 82.9 | 82.9 | 59.6 | の | be, the, they |
| dev-senpai | dev | en->ja | 先輩 | hard | gap | 13.2 | 63.9 | 63.9 | 77.4 | 友人 | mentor, fellow, colleague |
| dev-the | dev | ja->en | the | hard | gap | 24.9 | 52.3 | 52.3 | 43.9 | a | の, 其の, ザ |
| dev-estar | dev | en->es | estar | hard | split | 49.6 | 31.7 | 49.6 | 58.3 | tener | be, stand, lie |
| dev-saber | dev | en->es | saber | hard | split | 44.3 | 35.1 | 44.3 | 79.3 | desconocer | learn, know, aware |

## Task C: directional pairs (split direction should score higher)

| pair | split | hard side | difficulty | easy side | difficulty | correct | baseline correct |
|---|---|---|---|---|---|---|---|
| for-por | test | en->es por | 11.3 | es->en for | 11.2 | True | True |
| for-para | test | en->es para | 9.7 | es->en for | 11.2 | False | True |
| decir-tell | test | es->en tell | 58.2 | en->es decir | 81.9 | False | True |
| hacer-make | test | es->en make | 34.4 | en->es hacer | 54.2 | False | False |
| give-kureru | test | en->ja 呉れる | 62.4 | ja->en give | 59.1 | True | True |
| give-ageru | test | en->ja 上げる | 51.6 | ja->en give | 59.1 | False | True |
| dar-kureru | test | es->ja 呉れる | 74.7 | ja->es dar | 55.0 | True | True |
| be-ser | dev | en->es ser | 96.0 | es->en be | 6.4 | True | True |

Test accuracy: 0.429 (baseline 0.857), n = 7

## FSI consistency

Overall gap for English speakers: Spanish -0.001, Japanese +0.238 (FSI: Spanish category I, Japanese category IV).
