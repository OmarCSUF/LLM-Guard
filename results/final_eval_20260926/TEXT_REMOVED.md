# Prompt texts removed from stored prediction files

Reason: W&I+LOCNESS (BEA-2019) sentences are licensed for non-commercial research use and may not be redistributed. Only the `text` key was removed from each row (scripts/tools/strip_text.py); ids, labels, decisions and scores are unchanged, and every analysis script reads only those. The SHA-256 values before removal match the files in the original workspace.

| File | Rows | Rows with `text` removed | SHA-256 before | SHA-256 after |
|---|---:|---:|---|---|
| `results/final_eval_20260926/learner_corrected_s42_predictions/locness_n.jsonl` | 986 | 986 | `b2eef6a325d8b669aadf05a36eac9cddb1edf0b932e24a1fd55bdedb3db8e2ea` | `e32960b0e1b1e85791bc3d63ae197566fb9a3977aeff225cd4dfa4f4c2e58adc` |
| `results/final_eval_20260926/learner_corrected_s42_predictions/wi_a.jsonl` | 2494 | 2494 | `2fb3c59dba3160f425d126cf81c8db31f43eedc26f013b0278b7d58de44ec9fc` | `50107ae08db9cb2ad744d39dea83cb46c70d42a1c5daef987722583ac41c388b` |
| `results/final_eval_20260926/learner_corrected_s42_predictions/wi_b.jsonl` | 2739 | 2739 | `858cbbdf19ed4c503ef61641694bb14515ede53d908ac050538bc43c4480a23a` | `d7503738db8e33e6ae3581bc61ca0efa544574d9b49bcc4c91d582b22400a6eb` |
| `results/final_eval_20260926/learner_corrected_s42_predictions/wi_c.jsonl` | 2527 | 2527 | `a2926d352025ca9ed485eed772e0def185d53e7e7b1709ad37be9a4f7ae84551` | `f92726a1505bdf2dd9e1833602a326a871a2688ba7e4c573d8bddccfc273c795` |
| `results/final_eval_20260926/learner_corrected_s43_predictions/locness_n.jsonl` | 986 | 986 | `a9fce0460e03785334c13ba38dc957c18c4df3b255a010cad34cf52e0cdc4980` | `81e86963c0117145f772aa9c09e41a76374a7527e146a02b2f61573770b77121` |
| `results/final_eval_20260926/learner_corrected_s43_predictions/wi_a.jsonl` | 2494 | 2494 | `830ee44054b48412f79ef9b240703ddccabed1327be029b5f70b616344ed8d0b` | `9d123f61f0b3c4997ee7d41b57705858cad32081c8b736e42089d47d17116999` |
| `results/final_eval_20260926/learner_corrected_s43_predictions/wi_b.jsonl` | 2739 | 2739 | `0224ad318a4a6f9001cad25d53b0975ce9f43573061f9814a5ca306f5bd023d9` | `615296a19204d99645d7d359120cab045d2d648eaa8b8460ea5fc3a05cf9c531` |
| `results/final_eval_20260926/learner_corrected_s43_predictions/wi_c.jsonl` | 2527 | 2527 | `57eb6d94731e52615e908c077a7938d24ae4cf319673db578f13c319fb9fb62b` | `390696fa9fd372f2036c82416de9d990a8c70a311df1913a51a7527c4cb0a543` |
| `results/final_eval_20260926/learner_corrected_s44_predictions/locness_n.jsonl` | 986 | 986 | `d3948bbfb528b7f1accaa5a930cd19c433aa4e614448996cb706e6169e1ec306` | `f854e5120f967bda76553d2a898433b30fe576f62762cf41eccd7d0d6490480a` |
| `results/final_eval_20260926/learner_corrected_s44_predictions/wi_a.jsonl` | 2494 | 2494 | `6ca0137df33836c04b510240afb09ac179abfd4cdcdb2bd74ea67099ed7e28b2` | `b585f0a8fb6091d233e228f445c85800ecebcb7aefb931bb8c77a00be2607327` |
| `results/final_eval_20260926/learner_corrected_s44_predictions/wi_b.jsonl` | 2739 | 2739 | `6c000b00695494a80f2555f2a638d7ef100847ac1923aa91f50c0aaec083a475` | `fe4921a1aa654637ab60338471b1cd4a5dba247e691360e291b582be9a3da3e9` |
| `results/final_eval_20260926/learner_corrected_s44_predictions/wi_c.jsonl` | 2527 | 2527 | `4efb639bfef5b6723f118788a7ccc77c68f11ccd03365886c57aac0b0232569a` | `e17c4afc6bbf8e5977e11996d6db79a6d8e0a1a6126d970e2f4ebd73e2dca24d` |
| `results/final_eval_20260926/learner_original_s42_predictions/locness_n.jsonl` | 986 | 986 | `562c966a188a0ce346ca2c37572bb6419bb1ce52fa4e5126ec7aaf1480abcf6c` | `e36b52d57fd2bb3cfd85288bf9edbb72644b4789229123a0d1a7dbbf703b6ee7` |
| `results/final_eval_20260926/learner_original_s42_predictions/wi_a.jsonl` | 2494 | 2494 | `d18356cb6730a808fa0ce500f32fa5c92ff7f6ce3952a4b11699a9f35e04c18b` | `727ee31df0a0212c3722fd54730073977b867945dc03da750a07076d1c0cd4de` |
| `results/final_eval_20260926/learner_original_s42_predictions/wi_b.jsonl` | 2739 | 2739 | `370dced25d9ff29bbedb96c6faad2958dc9a54c5518d5787c696ce1783fed375` | `638ef75d75c4216d9bab73669e0d9fa43a4a360d774c5c4d60c4ce0c2c35a7c6` |
| `results/final_eval_20260926/learner_original_s42_predictions/wi_c.jsonl` | 2527 | 2527 | `fedff357326b6429521da196b7f7d9fd30a3b34e9f717bd1561e704180caf351` | `e213ab47f0cadfded1c0dfe33e47957d0f2ec0ad368467048c62400dcada42ea` |
| `results/final_eval_20260926/learner_original_s43_predictions/locness_n.jsonl` | 986 | 986 | `03f35e21f8e498bbaa46e0142d353ac4e7fe41294dc05b0c0811dea81bf5c7cb` | `e63170603ff4e3aa630b2db418ada71a082fb3fc19edd2dbcd3688ec4c62b360` |
| `results/final_eval_20260926/learner_original_s43_predictions/wi_a.jsonl` | 2494 | 2494 | `9da172e6eea22b7e26a525329753593ec2e9e5f921b1f6b04a3a917683764125` | `6042b6bb234ba212611f4f8032960e55f82aee0747c8a461b78eb2551c96a43d` |
| `results/final_eval_20260926/learner_original_s43_predictions/wi_b.jsonl` | 2739 | 2739 | `000f7b5d732efdf2a9511c07540885fa52dbb1552c83339f835d51fa7128daaf` | `ec60c51e7fd1b79b00500e1429e60517dd0c69b037e77a67dd82cc6bca3fc4f6` |
| `results/final_eval_20260926/learner_original_s43_predictions/wi_c.jsonl` | 2527 | 2527 | `d510c4e78bc6e9c27fe2f8abd3ccda2bd27f42f8fd5085c2639fa52eab0e5257` | `68e8360c3cb4d92cc63385b9aea4875620e442d2557083d98cba248baa72b660` |
| `results/final_eval_20260926/learner_original_s44_predictions/locness_n.jsonl` | 986 | 986 | `e7e5636b3fef7ff386ec540c4403252b48c11a8b25f9874ec0fd9c3c96a81b00` | `73044f95625e91955c6ad6d831c6449228f86f17982138f7dd0e2e38958d0870` |
| `results/final_eval_20260926/learner_original_s44_predictions/wi_a.jsonl` | 2494 | 2494 | `649f8e2462b6a56beff8b3056aa58a731decfbc7842e928708d66682d006331f` | `377937102ca699f3b726cbfaef09c4567d94174f448d58a4a52a10422e7b68b9` |
| `results/final_eval_20260926/learner_original_s44_predictions/wi_b.jsonl` | 2739 | 2739 | `e7ee9127b7eaf2487223066050409b5d00d0aee3bd47f367123364657f753767` | `1ea1d5661980390c0771e3e3b905b5187dadca4066c9f39df3bfd87da6b6eac7` |
| `results/final_eval_20260926/learner_original_s44_predictions/wi_c.jsonl` | 2527 | 2527 | `4d6f455461aa924c6e420a3e629f4e917ebbae142370aebcb4887014bc2740ef` | `f4f711b7ad211760ddacd962cc2ca965f471e6ad9fd9f67a3c0e2d3a13422c7c` |

| File | Rows | Rows with `text` removed | SHA-256 before | SHA-256 after |
|---|---:|---:|---|---|
| `results/final_eval_20260926/val_cascade_predictions/val.jsonl` | 1912 | 1912 | `8f50a8cc1f6930555b9da9913f6fb669e25b246b46efb184332d2f19c2404ad2` | `a5e16025e0ea64fe5a729df017f9da7cb8270c68355251c917e4f9e77bb0634c` |
| `results/final_eval_20260926/val_granite_predictions/val.jsonl` | 1912 | 1912 | `315de45a4581836c174c2151bce9d4cef291261dd5fa51b26af696c24db55e01` | `01881c93f63032c31211112050f11c04396b75b9f1977af2eff9eeb41aa43aae` |
| `results/final_eval_20260926/val_llamaguard_predictions/val.jsonl` | 1912 | 1912 | `49b9586d79762c14a01377cbbe18aa801920ffc9b126d05b5102199f467797f4` | `4a52b3d0b7a8d90ffc5b7bf30a91a4f9480b25d8a7f5f0c055131723982e487d` |
| `results/final_eval_20260926/val_shieldgemma_predictions/val.jsonl` | 1912 | 1912 | `4e594e6231ab9cb9ddf3133cc308ec4508e58347e9fb5cce1e75ed7781d57a9e` | `858abcd18f3364af2464957cf033aa2316e93081cba209a6c39a67117a0ae517` |
| `results/final_eval_20260926/pooltest_layer1_predictions/test.jsonl` | 1912 | 1912 | `c016d713bc96b7dfc7682ffea7a62fea3c4ad8de2c030959047dc3ffd652b939` | `bba9fc23d4edc5c5b085f173a6575f85938f2e00e1d91a7ba1a74f1f332b83dc` |

