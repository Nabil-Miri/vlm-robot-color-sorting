#!/usr/bin/env python3
"""
Vision-Language Color Sorting Demonstration

This script demonstrates automated color sorting using the Panda robot and a vision-language model (CLIP).
The robot identifies colored objects using CLIP and sorts them into matching colored zones.

Features:
- 5+ colored objects (e.g. red cube, blue cube, blue sphere, green sphere, yellow cylinder)
- 4 colored sorting zones (2 on left, 2 on right of robot)
- Vision-language (CLIP) based object identification and selection
- Automated color matching and placement
- Interactive text prompt mode for custom selection
- Visual feedback showing sorting progress
- Complete pick-and-place operations for each object

Usage:
    python3 color_sorting_vlm.py
"""

from panda_vision_simulation import VisionLanguagePandaSimulation
import pybullet as p
import time
import matplotlib.pyplot as plt
import torch


def main():
    """
    Run the color sorting demonstration.
    """
    print("🎨 Color Sorting Demonstration")
    print("=" * 40)
    print("Choose your demo mode:")
    print("1. 🎨 Automatic Color Sorting (sort all objects by color)")
    print("2. 🎯 Interactive Text Prompts (enter custom prompts)")
    print()
    
    choice = input("Select mode (1/2, default=1): ").strip()
    
    print("\nThe robot will work with these objects and zones:")
    print("• Objects: Red cube, blue cube, green sphere, yellow cylinder")
    print("• Zones: Red zone (left front), Blue zone (left back)")
    print("         Green zone (right front), Yellow zone (right back)")
    print()
    
    # Create simulation (using CLIP model for fast processing)
    sim = VisionLanguagePandaSimulation(
        gui_mode=True, 
        model_name="openai/clip-vit-base-patch32"
    )
    
    try:
        # Setup simulation environment with sorting zones
        print("Setting up simulation environment...")
        sim.setup_simulation()
        
        # Wait for physics to settle
        for _ in range(240):
            p.stepSimulation()
            time.sleep(1./240.)
        
        # Set robot to initial pose
        sim.set_initial_robot_pose()
        
        # Show initial scene
        print("\nShowing initial scene with objects and sorting zones...")
        initial_image = sim.capture_camera_image()
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(initial_image)
        ax.set_title("Initial Scene - Objects and Color Sorting Zones")
        ax.axis('off')
        
        # Add annotations for sorting zones
        zone_annotations = [
            "RED ZONE\n(right front)",
            "BLUE ZONE\n(left front)", 
            "GREEN ZONE\n(right back)",
            "YELLOW ZONE\n(left back)"
        ]
        
        # Add text annotations (approximate positions)
        ax.text(50, 50, "SORTING ZONES:", fontsize=14, weight='bold', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        for i, annotation in enumerate(zone_annotations):
            ax.text(50, 100 + i*30, f"• {annotation}", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
        
        plt.tight_layout()
        plt.show()
        
        # Run selected demo mode
        if choice == "2":
            # Interactive text prompts only
            run_interactive_text_prompt_demo(sim)
        else:
            # Automatic color sorting (default)
            print("\n🎨 Starting automatic color sorting...")
            results = sim.run_color_sorting_demo(visualize=True)
            
            # Final summary for auto mode
            successful_sorts = sum(1 for r in results if r['success'])
            print(f"\n{'='*50}")
            print(f"🎉 COLOR SORTING COMPLETE!")
            print(f"{'='*50}")
            print(f"Successfully sorted: {successful_sorts}/{len(results)} objects")
            
            if successful_sorts == len(results):
                print("🏆 Perfect sorting! All objects placed in correct zones.")
            else:
                print("⚠️  Some objects could not be sorted correctly.")
        
        print("\nPress Enter to exit...")
        input()
        
    except KeyboardInterrupt:
        print("\nColor sorting interrupted by user.")
    except Exception as e:
        print(f"Error during color sorting: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sim.cleanup()


def run_interactive_text_prompt_demo(sim):
    """
    Run an interactive demo where the user can enter text prompts
    to select and manipulate objects.
    
    Args:
        sim: VisionLanguagePandaSimulation instance
    """
    print(f"\n{'='*60}")
    print(f"🎯 INTERACTIVE TEXT PROMPT DEMO")
    print(f"{'='*60}")
    print("Enter text prompts to select objects for the robot to pick up!")
    print("Available objects: red_cube, blue_cube, green_sphere, yellow_cylinder")
    print("Example prompts:")
    print("  • 'red cube'")
    print("  • 'blue object'") 
    print("  • 'round thing'")
    print("  • 'yellow cylinder'")
    print("  • 'something green'")
    print("  • 'pick up the color with most amount of objects' (try full scene!)")
    print()
    print("ANALYSIS MODES:")
    print("• Type 'full:' before your prompt to use full scene analysis")
    print("• Example: 'full: pick the color with most objects'")
    print("• Regular prompts use individual object analysis")
    print()
    print("Type 'quit' or 'exit' to stop")
    print(f"{'='*60}")
    
    while True:
        try:
            # Get user input
            prompt = input("\n🤖 Enter your prompt: ").strip()
            
            # Check for exit commands
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Exiting interactive demo...")
                break
            
            # Skip empty prompts
            if not prompt:
                print("⚠️  Please enter a valid prompt!")
                continue
            
            print(f"\n🔍 Processing prompt: '{prompt}'")
            
            # Check if user wants full scene analysis
            # use_full_scene = prompt.lower().startswith('full:')
            # if use_full_scene:
            #     actual_prompt = prompt[5:].strip()  # Remove 'full:' prefix
            #     print(f"🌟 Using FULL SCENE analysis for: '{actual_prompt}'")
            #     
            #     # Analyze full scene
            #     scene_analysis = analyze_full_scene_with_prompt(sim, actual_prompt)
            #     
            #     print(f"\n📋 Full Scene Analysis Results:")
            #     print(f"   Method: {scene_analysis['method']}")
            #     print(f"   Analysis: {scene_analysis['analysis']}")
            #     
            #     if scene_analysis.get('scene_scores'):
            #         print(f"\n📊 Scene-based Scores:")
            #         for obj_name, score in scene_analysis['scene_scores'].items():
            #             bar = "█" * int(score * 20)
            #             print(f"     {obj_name:15}: {score:.3f} {bar}")
            #     
            #     recommended_object = scene_analysis['recommended_object']
            #     confidence = scene_analysis['confidence']
            #     
            #     if recommended_object:
            #         print(f"\n🎯 Full Scene Recommendation: {recommended_object} (confidence: {confidence:.3f})")
            #         
            #         # Ask if user wants to proceed with recommendation
            #         proceed = input(f"Use full scene recommendation '{recommended_object}'? (y/n/compare): ").lower()
            #         
            #         if proceed == 'compare':
            #             print(f"\n🔄 COMPARISON MODE: Full Scene vs Individual Crops")
            #             print("=" * 50)
            #             
            #             # Show both analyses
            #             print(f"🌟 Full Scene Result: {recommended_object} ({confidence:.3f})")
            #             
            #             # Now do individual crop analysis for comparison using segmentation
            #             print(f"\n🔍 Individual Crop Analysis for comparison (with segmentation)...")
            #             comparison_camera_image, comparison_bboxes, comparison_crops = sim.capture_and_process_scene()
            #             crop_scores = sim.compute_object_similarity(comparison_crops, actual_prompt)
            #             
            #             print(f"\n📊 Individual Crop Scores:")
            #             sorted_scores = sorted(crop_scores.items(), key=lambda x: x[1], reverse=True)
            #             for obj_name, score in sorted_scores:
            #                 bar = "█" * int(score * 20)
            #                 print(f"     {obj_name:15}: {score:.3f} {bar}")
            #             
            #             crop_winner, crop_score = sim.select_best_object(crop_scores)
            #             print(f"\n� Individual Crop Winner: {crop_winner} ({crop_score:.3f})")
            #             
            #             print(f"\n🤔 COMPARISON SUMMARY:")
            #             print(f"   Full Scene:      {recommended_object} ({confidence:.3f})")
            #             print(f"   Individual Crop: {crop_winner} ({crop_score:.3f})")
            #             
            #             if recommended_object == crop_winner:
            #                 print(f"   ✅ Both methods agree!")
            #             else:
            #                 print(f"   ⚠️  Methods disagree - which shows the difference!")
            #             
            #             # Let user choose which result to use
            #             choice = input(f"\nWhich result to use? (f=full scene, c=crop, n=none): ").lower()
            #             if choice == 'f':
            #                 selected_object, best_score = recommended_object, confidence
            #             elif choice == 'c':
            #                 selected_object, best_score = crop_winner, crop_score
            #             else:
            #                 print("🚫 Operation cancelled.")
            #                 continue
            #                 
            #         elif proceed in ['y', 'yes', '']:
            #             selected_object, best_score = recommended_object, confidence
            #         else:
            #             print("🚫 Operation cancelled.")
            #             continue
            #     else:
            #         print("❌ Full scene analysis couldn't determine a good object!")
            #         continue
                    
            # Regular individual crop analysis with improved segmentation
            print("🔍 Using INDIVIDUAL CROP analysis with segmentation masks")
            
            # Use new segmentation-based capture and processing
            print("📸 Capturing scene with segmentation...")
            camera_image, bounding_boxes, crops = sim.capture_and_process_scene()
            
            if not bounding_boxes:
                print("❌ No objects detected in scene!")
                continue
            
            # Compute similarity scores using improved masked crops
            similarity_scores = sim.compute_object_similarity(crops, prompt)
            
            if not similarity_scores:
                print("❌ Could not compute similarity scores!")
                continue
            
            # Show similarity results
            print("\n📊 Individual Crop Similarity Scores:")
            sorted_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
            for obj_name, score in sorted_scores:
                bar = "█" * int(score * 20)  # Visual bar
                print(f"  {obj_name:15}: {score:.3f} {bar}")
            
            # Select best object
            selected_object, best_score = sim.select_best_object(similarity_scores)
            
            if not selected_object:
                print("❌ No suitable object found!")
                continue
            
            print(f"\n🎯 Selected: {selected_object} (confidence: {best_score:.3f})")
            
            # Ask user for confirmation
            print("Options:")
            print("  y = Pick up and choose placement zone")
            print("  n = Cancel")
            confirm = input(f"What to do with {selected_object}? (y/n): ").lower()
            
            if confirm in ['y', 'yes', '']:
                print(f"🤖 Picking up {selected_object}...")
                success = sim.pick_object(selected_object)
                if success:
                    print(f"📍 Where would you like to place {selected_object}?")
                    print("1. 🔴 Red zone (left front)")
                    print("2. 🔵 Blue zone (left back)")
                    print("3. 🟢 Green zone (right front)")
                    print("4. 🟡 Yellow zone (right back)")
                    print("5. 🗑️  Throw away (fast throw!)")
                    
                    place_choice = input("Choose destination (1-5): ").strip()
                    
                    if place_choice == "1":
                        place_object_in_specific_zone(sim, selected_object, 'red_zone')
                        print(f"✅ Successfully placed {selected_object} in RED zone!")
                    elif place_choice == "2":
                        place_object_in_specific_zone(sim, selected_object, 'blue_zone')
                        print(f"✅ Successfully placed {selected_object} in BLUE zone!")
                    elif place_choice == "3":
                        place_object_in_specific_zone(sim, selected_object, 'green_zone')
                        print(f"✅ Successfully placed {selected_object} in GREEN zone!")
                    elif place_choice == "4":
                        place_object_in_specific_zone(sim, selected_object, 'yellow_zone')
                        print(f"✅ Successfully placed {selected_object} in YELLOW zone!")
                    elif place_choice == "5":
                        throw_object_away(sim, selected_object)
                        print(f"🗑️ Threw away {selected_object} with style!")
                    else:
                        print("❌ Invalid choice, using safe placement...")
                        safe_place_object(sim, selected_object)
                        print(f"✅ Safely placed {selected_object}!")
                else:
                    print(f"❌ Failed to pick up {selected_object}")
            else:
                print("🚫 Operation cancelled.")
            
            # Show updated scene
            print("\n📸 Updated scene:")
            final_image = sim.capture_camera_image()
            
            # Optional: show images side by side
            show_images = input("Show before/after images? (y/n): ").lower()
            if show_images in ['y', 'yes']:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                ax1.imshow(camera_image)
                ax1.set_title("Before")
                ax1.axis('off')
                
                ax2.imshow(final_image)
                ax2.set_title("After")
                ax2.axis('off')
                
                plt.tight_layout()
                plt.show()
                
        except KeyboardInterrupt:
            print("\n👋 Demo interrupted by user.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("Interactive demo completed!")


def analyze_full_scene_with_prompt(sim, prompt):
    """
    Analyze the full scene image with the prompt instead of individual crops.
    This gives CLIP the complete context to understand spatial relationships
    and potentially count objects.
    
    Args:
        sim: VisionLanguagePandaSimulation instance
        prompt: Text prompt to analyze against full scene
        
    Returns:
        dict: Analysis results including scene description and object recommendations
    """
    print(f"\n🌟 FULL SCENE ANALYSIS MODE")
    print("=" * 40)
    print("Analyzing the complete scene with AI vision...")
    
    # Capture the full scene
    camera_image = sim.capture_camera_image()
    
    # Display the input image (commented out to avoid showing intermediate photos)
    # print("📸 Displaying input scene for analysis...")
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    # ax.imshow(camera_image)
    # ax.set_title(f"Full Scene Input for Analysis\nPrompt: '{prompt}'")
    # ax.axis('off')
    
    # Add bounding boxes to show detected objects (commented out)
    # bounding_boxes = sim.get_object_bounding_boxes(camera_image)
    # colors = plt.cm.Set3(range(len(bounding_boxes)))
    
    # for i, (obj_name, bbox) in enumerate(bounding_boxes.items()):
    #     x1, y1, x2, y2 = bbox
    #     color = colors[i]
    #     
    #     # Draw bounding box
    #     import matplotlib.patches as patches
    #     rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
    #                            linewidth=2, edgecolor=color, 
    #                            facecolor='none', alpha=0.8)
    #     ax.add_patch(rect)
    #     
    #     # Add label
    #     ax.text(x1, y1-5, obj_name, fontsize=10, 
    #             bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
    
    # plt.tight_layout()
    # plt.show()
    
    # Use CLIP for full scene analysis
    return _analyze_full_scene_clip(sim, camera_image, prompt)


def _analyze_full_scene_clip(sim, camera_image, prompt):
    """Analyze full scene using CLIP model with improved approach."""
    
    print("🔍 Step 1: Analyzing scene context with CLIP...")
    
    # First, let's understand what the prompt is asking for
    # by analyzing it against scene-level descriptions
    scene_understanding_prompts = [
        "a scene with multiple red objects",
        "a scene with multiple colored objects", 
        "objects grouped by color",
        "red objects are the most numerous",
        "green objects are the most numerous",
        "yellow objects are the most numerous",
        prompt  # The original prompt
    ]
    
    # Analyze scene understanding
    inputs = sim.processor(
        text=scene_understanding_prompts,
        images=[camera_image] * len(scene_understanding_prompts),
        return_tensors="pt",
        padding=True
    ).to(sim.device)
    
    with torch.no_grad():
        outputs = sim.model(**inputs)
        scene_logits = outputs.logits_per_image[0]
        scene_probs = torch.softmax(scene_logits, dim=0)
    
    print("📊 Scene Understanding Scores:")
    for i, scene_prompt in enumerate(scene_understanding_prompts):
        score = scene_probs[i].item()
        print(f"   '{scene_prompt}': {score:.3f}")
    
    # Step 2: Now analyze each object individually against the ORIGINAL prompt
    print(f"\n🔍 Step 2: Analyzing each object type against the prompt...")
    
    # Get available objects using new segmentation approach
    print(f"\n🔍 Step 2: Using segmentation-based object detection...")
    _, bounding_boxes, crops = sim.capture_and_process_scene()
    available_objects = list(bounding_boxes.keys())
    
    print(f"   Available objects: {available_objects}")
    
    if not available_objects:
        print("   ❌ No objects detected!")
        return {
            'method': 'full_scene_clip_improved',
            'scene_scores': {},
            'recommended_object': None,
            'confidence': 0.0,
            'analysis': "No objects detected in scene"
        }
    
    print(f"   Processing {len(available_objects)} objects with CLIP...")
    
    # Create specific prompts for each object to better match against the user prompt
    object_specific_prompts = []
    for obj_name in available_objects:
        # Create a contextual prompt that includes the object name and user's intent
        if "most" in prompt.lower() and "object" in prompt.lower():
            # For counting queries, use a comparative prompt
            object_specific_prompts.append(f"the {obj_name.replace('_', ' ')} when asked to {prompt}")
        else:
            # For direct selection, use descriptive prompt
            object_specific_prompts.append(f"a {obj_name.replace('_', ' ')} that matches: {prompt}")
    
    print(f"   Object-specific prompts: {object_specific_prompts}")
    
    # Process with CLIP: we want text-image similarity for each object prompt against the scene
    inputs = sim.processor(
        text=[prompt],  # Single user prompt
        images=[camera_image],  # Single scene image
        return_tensors="pt",
        padding=True
    ).to(sim.device)
    
    print(f"   Input tensor shapes: text=1, images=1")
    
    # Use improved crops from segmentation
    scene_scores = {}
    
    # Analyze each object crop against the user prompt using segmentation-based crops
    for obj_name in available_objects:
        if obj_name in crops:
            # Process this specific segmentation-based crop with the user prompt
            crop_inputs = sim.processor(
                text=[prompt],
                images=[crops[obj_name]],
                return_tensors="pt",
                padding=True
            ).to(sim.device)
            
            with torch.no_grad():
                crop_outputs = sim.model(**crop_inputs)
                # Get similarity score between the prompt and this object crop
                similarity_score = crop_outputs.logits_per_image[0, 0].item()
                # Convert logit to probability-like score
                scene_scores[obj_name] = torch.sigmoid(torch.tensor(similarity_score)).item()
        else:
            scene_scores[obj_name] = 0.0
            print(f"   Warning: No segmentation crop available for {obj_name}")
    
    print(f"   Computed {len(scene_scores)} individual object scores")
    
    print("📊 Object-Level Full Scene Scores:")
    for obj_name, score in scene_scores.items():
        print(f"   {obj_name}: {score:.3f}")
    
    # Step 3: Apply domain knowledge for counting-based prompts
    if "most" in prompt.lower() and ("amount" in prompt.lower() or "objects" in prompt.lower()):
        print("\n🧮 Step 3: Applying counting logic...")
        
        # Count objects by color
        color_counts = {}
        for obj_name in available_objects:
            if 'red' in obj_name.lower():
                color_counts['red'] = color_counts.get('red', 0) + 1
            elif 'blue' in obj_name.lower():
                color_counts['blue'] = color_counts.get('blue', 0) + 1
            elif 'green' in obj_name.lower():
                color_counts['green'] = color_counts.get('green', 0) + 1
            elif 'yellow' in obj_name.lower():
                color_counts['yellow'] = color_counts.get('yellow', 0) + 1
        
        print(f"   Color counts: {color_counts}")
        
        # Find most common color
        if color_counts:
            most_common_color = max(color_counts.items(), key=lambda x: x[1])
            print(f"   Most common color: {most_common_color[0]} ({most_common_color[1]} objects)")
            
            # Boost scores for objects of the most common color
            boosted_scores = scene_scores.copy()
            boost_applied = False
            for obj_name in scene_scores:
                if most_common_color[0] in obj_name.lower():
                    # Significantly boost confidence for correct color
                    original_score = scene_scores[obj_name]
                    boosted_scores[obj_name] = min(0.95, original_score + 0.4)
                    print(f"   Boosted {obj_name}: {original_score:.3f} → {boosted_scores[obj_name]:.3f}")
                    boost_applied = True
            
            if boost_applied:
                print(f"   Applied boost to {most_common_color[0]} objects")
                scene_scores = boosted_scores
            else:
                print(f"   Warning: No objects found matching most common color '{most_common_color[0]}'")
            
            # Debug: show final scores after boosting
            print("📊 Final scores after boosting:")
            for obj_name, score in scene_scores.items():
                print(f"   {obj_name}: {score:.3f}")
    
    # Find best match
    if scene_scores:
        # Sort by score to find the best match
        sorted_scores = sorted(scene_scores.items(), key=lambda x: x[1], reverse=True)
        best_object = sorted_scores[0]
        
        print(f"\n🏆 Final ranking:")
        for i, (obj_name, score) in enumerate(sorted_scores):
            rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📍"
            print(f"   {rank_emoji} {obj_name}: {score:.3f}")
        
        return {
            'method': 'full_scene_clip_improved',
            'scene_scores': scene_scores,
            'recommended_object': best_object[0],
            'confidence': best_object[1],
            'analysis': f"Improved full scene analysis: {best_object[0]} (confidence: {best_object[1]:.3f})"
        }
    else:
        return {
            'method': 'full_scene_clip_improved',
            'scene_scores': {},
            'recommended_object': None,
            'confidence': 0.0,
            'analysis': "Could not analyze full scene"
        }


def place_object_in_specific_zone(sim, object_name, zone_name):
    """
    Place the currently held object in a specific color sorting zone.
    
    Args:
        sim: VisionLanguagePandaSimulation instance
        object_name (str): Name of the object being placed
        zone_name (str): Name of the target zone ('red_zone', 'blue_zone', etc.)
    """
    if zone_name not in sim.sorting_zones:
        print(f"❌ Unknown zone: {zone_name}")
        safe_place_object(sim, object_name)  # Fallback to safe position
        return
    
    # Get target zone position
    zone_config = sim.sorting_zones[zone_name]
    zone_position = zone_config['position']
    zone_color = zone_name.replace('_zone', '').upper()
    
    print(f"   Placing {object_name} in {zone_color} zone at {zone_position}")
    
    # Move safely to zone approach position (higher up to avoid table collision)
    zone_approach = [zone_position[0], zone_position[1], zone_position[2] + 0.25] 
    print(f"   Moving safely to approach position...")
    sim.safe_move_to_position(zone_approach)
    
    # Lower to zone surface (just above the colored zone)
    zone_target = [zone_position[0], zone_position[1], zone_position[2] + 0.05]  # Higher above zone
    sim.move_to_position(zone_target)
    
    # Release object
    print(f"   Releasing {object_name} in {zone_color} zone...")
    sim.control_gripper(sim.gripper_open_position)
    
    # Retreat with extra height using safe movement
    retreat_pos = [zone_position[0], zone_position[1], zone_position[2] + 0.25]  
    print(f"   Retreating safely...")
    sim.move_to_position(retreat_pos)
    
    # Return to home
    print("   Returning to home position...")
    sim.move_to_position(sim.home_position)


def safe_place_object(sim, object_name):
    """
    Safely place object with higher trajectory to avoid table collisions.
    
    Args:
        sim: VisionLanguagePandaSimulation instance
        object_name (str): Name of the object being placed
    """
    print(f"   Safely placing {object_name}...")
    
    # Move to safe approach position (much higher) using safe movement
    safe_approach = [sim.place_position[0], sim.place_position[1], sim.place_position[2] + 0.30] 
    print(f"   Moving safely to approach position...")
    sim.safe_move_to_position(safe_approach)
    
    # Lower to safe place position (higher than original)
    safe_target = [sim.place_position[0], sim.place_position[1], sim.place_position[2] + 0.05]
    sim.move_to_position(safe_target)
    
    # Release object
    print(f"   Releasing {object_name} safely...")
    sim.control_gripper(sim.gripper_open_position)
    
    # Retreat with extra height
    safe_retreat = [sim.place_position[0], sim.place_position[1], sim.place_position[2] + 0.30] 
    print(f"   Retreating safely...")
    sim.move_to_position(safe_retreat)
    
    # Return to home
    print("   Returning to home position...")
    sim.move_to_position(sim.home_position)


def throw_object_away(sim, object_name):
    """
    Throw the object away with a fast motion and dramatic release!
    
    Args:
        sim: VisionLanguagePandaSimulation instance
        object_name (str): Name of the object being thrown
    """
    print(f"🗑️ Throwing away {object_name} with style!")
    
    # Define throw position (to the side, away from workspace)
    throw_position = [-0.3, 0.5, 0.8]  # To the left and forward, high up
    
    # Fast approach to throw position using safe movement
    print("   🚀 Moving safely to throw position...")
    approach_pos = [throw_position[0], throw_position[1], throw_position[2] + 0.1]
    
    # Use safe movement to get to throw position without hitting table
    sim.safe_move_to_position(approach_pos)
    
    # DRAMATIC RELEASE!
    print("   💨 RELEASING WITH STYLE!")
    sim.control_gripper(sim.gripper_open_position)
    
    # Quick retreat
    print("   ↩️  Quick retreat...")
    retreat_pos = [0.2, 0.0, 0.9]  # Quick retreat position
    sim.move_to_position(retreat_pos)
    
    # Return to home with satisfaction
    print("   🏠 Mission accomplished, returning home...")
    sim.move_to_position(sim.home_position)


if __name__ == "__main__":
    main()
