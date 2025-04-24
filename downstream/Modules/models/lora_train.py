def main(args):
    # Setup TensorBoard logging
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load base model
    print(f"Loading base model from {args.model_path}")
    model = EEGPT_calibry(
        n_chans=args.n_channels,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
        encoder_dim=args.encoder_dim,
        decoder_dim=args.decoder_dim,
        encoder_heads=args.encoder_heads,
        decoder_heads=args.decoder_heads,
        ckpt_path=args.model_path,
        freeze_encoder=args.freeze_encoder,
        out_dim=args.out_dim
    )
    model = model.to(device)
    
    # Set up LoRA if enabled
    if args.use_lora:
        print("Using LoRA for efficient fine-tuning")
        config = LoraConfig(
            r=args.lora_r, 
            lora_alpha=args.lora_alpha, 
            target_modules=["q", "k", "v", "o", "fc"], 
            lora_dropout=args.lora_dropout, 
            bias="none",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    
    # Prepare datasets and dataloaders
    print(f"Loading data from {args.data_dir}")
    
    # Prepare train, val, test datasets
    train_dataset, val_dataset, test_dataset = prepare_p300_data(
        subject=args.subject,
        dataset_fold=args.data_dir,
        ch_names=None,
        target_count=args.samples_per_class,
        nontarget_count=args.samples_per_class,
        max_time_length=args.max_time_length,
        model_patch_size=args.patch_size,
        paradigm=args.paradigm,
        sfreq=args.sampling_rate,
        seed=args.seed,
        model_chans_count=args.n_channels
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if args.use_lora:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Train and evaluate
    best_accuracy = 0.0
    best_model_path = os.path.join(args.output_dir, f"best_model_{args.subject}.pth")
    
    # Training loop
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': train_loss / (progress_bar.n + 1),
                'acc': 100. * train_correct / train_total
            })
        
        # Calculate average training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100. * val_correct / val_total
        
        # Update scheduler
        scheduler.step(val_accuracy)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            
            # Create output directory if it doesn't exist
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Save model
            if args.use_lora:
                model.save_pretrained(best_model_path)
            else:
                torch.save(model.state_dict(), best_model_path)
            
            print(f"  New best model saved with accuracy: {best_accuracy:.2f}%")
    
    # Close TensorBoard writer
    writer.close()
    
    # Test with best model
    print("\nEvaluating best model on test set...")
    
    # Load best model
    if args.use_lora:
        model = PeftModel.from_pretrained(model, best_model_path)
    else:
        model.load_state_dict(torch.load(best_model_path))
    
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Statistics
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    # Calculate test accuracy
    test_accuracy = 100. * test_correct / test_total
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # Save final results
    results = {
        'subject': args.subject,
        'best_val_accuracy': best_accuracy,
        'test_accuracy': test_accuracy,
        'model_params': {
            'n_channels': args.n_channels,
            'patch_size': args.patch_size,
            'use_lora': args.use_lora,
            'lora_r': args.lora_r if args.use_lora else None
        }
    }
    
    # Save results to JSON
    results_path = os.path.join(args.output_dir, f"results_{args.subject}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {results_path}")
    print("Training completed!") 