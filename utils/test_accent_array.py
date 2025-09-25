from ruaccent import RUAccent

# Test data - array of strings to accentize
test_data = [
    "\"Игра на выживание\" - это 18 человек",
    "из самых разных слоев общества.",
    "Они все собираются здесь,",
    "им приходится доверять друг другу, объединяться, проходить испытания,",
    "у них самые разнообразные сильные и слабые стороны.",
    "Им нужно доверять друг другу, но в то же время не доверять никому,",
    "ведь эта игра построена на обмане.",
    "Необходимо быть уязвимым,",
    "а также быть самим собой.",
    "Нужно преодолевать и адаптироваться",
    "ко всем предстоящим испытаниям,",
    "ведь в этом и заключается \"Игра на выживание\".",
    "Но именно из-за этого эта игра стоит свеч.",
    "И я к ней готов.",
    "Бану, Яну Я подал заявку на участие в тот день, когда стал гражданином США.",
    "Я вышел из здания суда и закричал: \"Игра на выживание\"! Джефф Пробст!\"",
    "Всем привет, намасте.",
    "Я хочу получить возможность принять участие в игре,",
    "чтобы стать образцом для подражания для всех, кто выглядит так же, как я.",
    "Из трущоб в \"Игру на выживание\"!"
]

def test_accent_array():
    # Initialize accent model
    accent = RUAccent()
    accent.load(omograph_model_size="turbo3.1", use_dictionary=False, device="CPU")
    
    try:
        # Process the array
        accented_data = accent.process_array(test_data)
        
        # Verify lengths match
        assert len(accented_data) == len(test_data), \
            f"Length mismatch! Input: {len(test_data)}, Output: {len(accented_data)}"
    
        print("Array length test passed - input and output lengths match")
        print("\nSample results:")
        for orig, accented in zip(test_data[:3], accented_data[:3]):
            print(f"Original: {orig}")
            print(f"Accented: {accented}")
            print()
            
    except Exception as e:
        print(f"Error processing array: {str(e)}")

if __name__ == "__main__":
    test_accent_array()