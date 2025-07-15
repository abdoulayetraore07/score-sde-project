#!/usr/bin/env python3
"""
Module commun pour modifier les configurations de sampling de façon interactive.
Réutilisable par pretrained_sampling.py et pretrained_controllable_gen.py
"""

USE_CUSTOM = False

def get_config_value(config, path):
    """Récupère une valeur depuis config avec gestion des défauts."""
    if "." in path:
        parts = path.split(".")
        obj = config
        for part in parts:
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj
    else:
        defaults = {
            "guidance_strategy": "standard"
        }
        return getattr(config, path, defaults.get(path, None))


def set_config_value(config, path, value):
    """Définit une valeur dans config."""
    if "." in path:
        parts = path.split(".")
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    else:
        setattr(config, path, value)


def get_overridable_fields():
    """Retourne le mapping complet des paramètres modifiables."""
    return {
        "sampling_method": "sampling.method",   
        "predictor": "sampling.predictor", 
        "corrector": "sampling.corrector",
        "snr": "sampling.snr",
        "denoise": "sampling.noise_removal",
        "n_steps": "sampling.n_steps_each",
        "probability_flow": "sampling.probability_flow",
        "guidance_scale": "guidance_scale", 
        "guidance_strategy": "guidance_strategy",
        "adaptive_sigma_limit": "adaptive_sigma_limit",
        "step_lr": "step_lr" 
    }

        #"sampling_h_init": "sampling.sampling_h_init",
        #"sampling_abstol": "sampling.sampling_abstol", 
        #"sampling_reltol": "sampling.sampling_reltol",
        #"error_use_prev": "sampling.error_use_prev",
        #"norm": "sampling.norm",
        #"sampling_safety": "sampling.sampling_safety",
        #"extrapolation": "sampling.extrapolation",
        #"sde_improved_euler": "sampling.sde_improved_euler",
        #"sampling_exp": "sampling.sampling_exp"

def get_guidance_strategy_options():
    """Retourne les options disponibles pour guidance_strategy."""
    return {
        "standard": "Baseline guidance (no adaptation)",
        "adaptive_scale": "Linear guidance scale variation",
        "truncation": "Sigma truncation method", 
        "amplification": "Artificial score amplification"
    }


def interactive_config_modification(config):
    """
    Interface interactive pour modifier la configuration.
    
    Args:
        config: Configuration à modifier
        
    Returns:
        bool: True si des modifications ont été faites, False sinon
    """
    if not USE_CUSTOM :
        return 
    
    OVERRIDABLE_FIELDS = get_overridable_fields()
    
    print("\n🔧 Paramètres configurables :")
    print("=" * 50)
    
    # Afficher tous les paramètres avec leurs valeurs actuelles
    for i, key in enumerate(OVERRIDABLE_FIELDS.keys(), 1):
        current_val = get_config_value(config, OVERRIDABLE_FIELDS[key])
        if current_val is not None:
            if key == "guidance_strategy":
                strategies = get_guidance_strategy_options()
                description = strategies.get(current_val, "Unknown")
                print(f"  {i:2d}. {key:<18} (actuel: {current_val} - {description})")
            else:
                print(f"  {i:2d}. {key:<18} (actuel: {current_val})")
        else:
            print(f"  {i:2d}. {key:<18} (non trouvé)")
    
    print(f"   0. Continuer sans modification")
    print("=" * 50)
    
    try:
        choice_input = input("\nEntrez le(s) numéro(s) à modifier (ex: 1,3,5 ou 0 pour continuer): ").strip()
        
        if choice_input == "0" or choice_input == "":
            print("📋 Aucune modification - configuration par défaut utilisée")
            return False
        
        # Parser les choix multiples
        try:
            choices = [int(x.strip()) for x in choice_input.split(",")]
        except ValueError:
            print("❌ Format invalide - utilisez des numéros séparés par des virgules")
            return False
        
        # Valider les choix
        valid_choices = []
        field_keys = list(OVERRIDABLE_FIELDS.keys())
        
        for choice in choices:
            if 1 <= choice <= len(field_keys):
                valid_choices.append(choice)
            else:
                print(f"⚠️ Choix {choice} invalide (doit être entre 1 et {len(field_keys)})")
        
        if not valid_choices:
            print("❌ Aucun choix valide - configuration par défaut utilisée")
            return False
        
        # Modifier les paramètres sélectionnés
        modifications_made = False
        
        for choice_num in valid_choices:
            selected_key = field_keys[choice_num - 1]
            config_path = OVERRIDABLE_FIELDS[selected_key]
            current_val = get_config_value(config, config_path)
            
            if current_val is None:
                print(f"⚠️ Impossible de trouver {selected_key} dans config, ignoré")
                continue
            
            print(f"\n🔧 Modification de '{selected_key}':")
            print(f"   Valeur actuelle: {current_val}")
            
            try:
                if selected_key == "guidance_strategy":
                    strategies = get_guidance_strategy_options()
                    print("   Options disponibles:")
                    for j, (key, desc) in enumerate(strategies.items(), 1):
                        marker = "★" if key == current_val else " "
                        print(f"    {j}. {key:<15} - {desc} {marker}")
                    
                    while True:
                        try:
                            strat_choice = input("   Choisissez une stratégie (1-4): ").strip()
                            if not strat_choice:
                                break
                            strat_idx = int(strat_choice) - 1
                            strategy_keys = list(strategies.keys())
                            if 0 <= strat_idx < len(strategy_keys):
                                new_val = strategy_keys[strat_idx]
                                break
                            else:
                                print("   ❌ Choix invalide, réessayez")
                        except ValueError:
                            print("   ❌ Entrez un numéro valide")
                
                elif isinstance(current_val, bool):
                    while True:
                        new_input = input(f"   Nouvelle valeur (True/False): ").strip().lower()
                        if not new_input:
                            break
                        if new_input in ['true', 't', '1', 'yes', 'y']:
                            new_val = True
                            break
                        elif new_input in ['false', 'f', '0', 'no', 'n']:
                            new_val = False
                            break
                        else:
                            print("   ❌ Entrez True ou False")
                            
                elif isinstance(current_val, (float, int)):
                    while True:
                        try:
                            new_input = input(f"   Nouvelle valeur: ").strip()
                            if not new_input:
                                break
                            new_val = float(new_input) if isinstance(current_val, float) else int(new_input)
                            break
                        except ValueError:
                            print("   ❌ Entrez un nombre valide")
                            
                else:  # String
                    new_input = input(f"   Nouvelle valeur: ").strip()
                    if not new_input:
                        continue
                    new_val = new_input
                
                # Appliquer le changement seulement si une nouvelle valeur a été donnée
                if 'new_val' in locals():
                    set_config_value(config, config_path, new_val)
                    print(f"   ✅ {selected_key}: {current_val} → {new_val}")
                    modifications_made = True
                    del new_val
                else:
                    print(f"   📋 {selected_key}: valeur inchangée")
                
            except KeyboardInterrupt:
                print(f"\n   ⏭️ {selected_key}: annulé, valeur inchangée")
                continue
            except Exception as e:
                print(f"   ❌ Erreur: {e} - valeur inchangée")
                continue
        
        if modifications_made:
            print(f"\n✅ {len([c for c in valid_choices if c])} paramètre(s) modifié(s)")
        
        return modifications_made
        
    except KeyboardInterrupt:
        print("\n📋 Modification annulée - configuration par défaut utilisée")
        return False
    except Exception as e:
        print(f"❌ Erreur inattendue: {e} - configuration par défaut utilisée")
        return False


def print_config_summary(config, model_name, num_samples, checkpoint_path, modified=False):
    """Affiche un résumé de la configuration."""
    
    print("=" * 80)
    if modified:
        print("CONFIGURATION MODIFIÉE - RÉSUMÉ")
    else:
        print("CONFIGURATION PAR DÉFAUT - RÉSUMÉ")
    print("=" * 80)
    print(f"🎯 Model: {model_name}")
    print(f"📁 Checkpoint: {checkpoint_path}")
    print(f"🖼️  Samples to generate: {num_samples}")
    print(f"📐 Resolution: {config.data.image_size}x{config.data.image_size}")
    print(f"🎨 Sampling method: {config.sampling.method}")
    print(f"📊 Predictor: {config.sampling.predictor}")
    print(f"🔧 Corrector: {config.sampling.corrector}")
    print(f"📈 SNR: {config.sampling.snr}")
    print(f"🧹 Denoising: {config.sampling.noise_removal}")
    print(f" Sigma_max : {config.model.sigma_max}")
    
    # Afficher guidance_scale si disponible
    guidance_scale = get_config_value(config, "guidance_scale")
    if guidance_scale is not None:
        print(f"🎯 Guidance scale: {guidance_scale}")
    
    # Afficher guidance_strategy si disponible
    guidance_strategy = get_config_value(config, "guidance_strategy")
    if guidance_strategy is not None:
        strategies = get_guidance_strategy_options()
        description = strategies.get(guidance_strategy, "Unknown")
        print(f"🚀 Guidance strategy: {guidance_strategy} - {description}")
        
        # Afficher adaptive_sigma_limit si stratégie adaptive_scale
        if guidance_strategy == "adaptive_scale":
            adaptive_sigma_limit = get_config_value(config, "adaptive_sigma_limit")
            if adaptive_sigma_limit is not None:
                print(f"📏 Adaptive sigma limit: {adaptive_sigma_limit}")
    
    print("=" * 80)
    print("")