"""Generate a small labelled SMS dataset (spam vs ham)."""
import os
import pandas as pd

ham = [
    "Hey, are we still meeting at 5pm today?",
    "Don't forget to pick up milk on the way home.",
    "Can you send me the notes from class?",
    "Dinner at my place tomorrow, bring dessert!",
    "The project deadline is moved to Friday.",
    "Just got home, call me when you're free.",
    "Your order has shipped and will arrive in 3 days.",
    "Happy birthday! Hope you have a great day.",
    "Let's go for a walk after lunch.",
    "Please review the attached document by tomorrow.",
    "I'll be there in 10 minutes.",
    "Thanks for helping out last weekend.",
    "Do you want coffee or tea?",
    "The wifi password is on the fridge.",
    "See you at the library at 3.",
    "Mom asked us to call her tonight.",
    "I finished the assignment early, want to check it?",
    "We're running late, start without us.",
    "Great job on the presentation today!",
    "Can you bring my charger to class?",
    "The weather looks nice, let's go out.",
    "My flight lands at 8 tonight.",
    "Don't reply, this is just a reminder.",
    "I booked tickets for the movie on Saturday.",
    "Are you coming to the gym today?",
    "The package is at the front desk.",
    "Lunch break at noon, see you there.",
    "I'll send the file once I'm on my laptop.",
    "Thanks, that makes sense now.",
    "Remind me to water the plants tomorrow.",
    "Game night at 7, bring snacks.",
    "The meeting got cancelled, we're free.",
    "Can you take a look at my code when free?",
    "I left my jacket at your place, can I grab it?",
    "We should study for the exam together.",
    "New cafe opened downtown, want to try it?",
    "I'll call you after my lecture ends.",
    "Your Uber will arrive in 2 minutes.",
    "The show was amazing, you missed out.",
    "Let me know when you're available.",
]

spam = [
    "Congratulations! You've won a free iPhone. Claim now: bit.ly/claim",
    "URGENT: Your account has been suspended. Verify immediately.",
    "Win $1000 cash today! Click here to enter the lottery.",
    "Get rich quick! Work from home and earn $500/day.",
    "FREE TRIAL of our miracle weight loss supplement. Limited time!",
    "You have been selected for a special prize. Call now!",
    "Limited offer: 90% off all products today only!",
    "Your parcel is waiting. Pay shipping to release it: link",
    "Act fast! 0% interest personal loans approved instantly.",
    "Hot singles near you want to chat. Click here!",
    "You've inherited millions from a distant relative. Send fees to claim.",
    "Guaranteed returns on your investment. Double your money in a week!",
    "Last chance to renew your extended warranty for your car.",
    "Exclusive VIP sale - prices slashed 80% for members only.",
    "Claim your free gift card now before it expires!",
    "Your bank needs you to verify your PIN. Reply with it now.",
    "Click this link to see who viewed your profile!",
    "Buy quality watches at 50% off - replica grade A.",
    "Earn money while you sleep - no experience needed!",
    "You've won a trip to the Bahamas! Confirm with your details.",
    "Missed call from unknown number? Find out who - install now.",
    "Special prize number 5000. Call 1-900-xxx to redeem.",
    "We have a job offer that pays cash weekly. No interview needed.",
    "Your order is delayed. Update payment method to reschedule.",
    "Free credits added to your account - withdraw today!",
    "Make $1000 by filling a simple survey. Sign up now.",
    "Secret recipe to melt belly fat fast - click to order.",
    "Sweepstakes entry confirmed. You may already be a winner!",
    "Lowest prices on meds without prescription. Order online.",
    "Your computer is infected! Call this number immediately.",
    "Become a mystery shopper - get paid to shop! Apply today.",
    "Unlock premium content absolutely free - no strings attached.",
    "Casino bonus match up to $5000. Sign up instantly.",
    "We noticed unusual activity. Login to secure your account.",
    "Cheap flights to Europe - book within 24 hours!",
    "Your subscription will auto-renew tomorrow. Cancel here.",
    "Instant approval for credit cards regardless of credit score.",
    "Diamond ring giveaway - participate now and win big.",
    "Update your app to remove viruses detected on your phone.",
    "Your refund of $2500 is pending. Confirm your bank details.",
]

h = [("ham", m) for m in ham]
s = [("spam", m) for m in spam]

import random
random.seed(2)

ham_templates = [
    "Can we reschedule {0} to {1}?",
    "I'm at {0}, are you coming?",
    "Reminder: {0} is due on {1}.",
    "Let's meet at {0} after class.",
    "Did you get my email about {0}?",
    "Call me when you're at {0}.",
    "Bring {0} with you tomorrow.",
    "See you at {1} sharp.",
    "I'll text you the {0} later.",
    "How was your {0} today?",
]
ham_words = ["the meeting", "lunch", "the gym", "the library", "home", "the cafe",
             "the park", "the lab", "dinner", "the game"]
ham_times = ["5pm", "noon", "3 o'clock", "7pm", "after work", "tomorrow morning", "tonight", "6pm"]

for i in range(320):
    t = random.choice(ham_templates).format(random.choice(ham_words), random.choice(ham_times))
    h.append(("ham", t))

spam_templates = [
    "FREE {0} - claim yours now at {1}.",
    "You have won a {0}! Redeem at {1} immediately.",
    "URGENT: your {0} is about to expire. Act now: {1}.",
    "Get {0} today for only $9.99 at {1}.",
    "Your {0} reward is waiting. Visit {1} now.",
    "WIN a {0} in our giveaway! Enter at {1}.",
    "Limited time: {0} sale up to 90% off at {1}.",
    "Confirm your {0} details to release payment at {1}.",
    "Earn extra cash with {0} - sign up at {1}.",
    "Your {0} gift card has been credited. Use at {1}.",
]
spam_nouns = ["vacation", "iPhone", "cash prize", "gift card", "discount", "lottery",
              "free trial", "refund", "bonus", "membership", "airline miles", "spa package"]
spam_links = ["bit.ly/claim", "tinyurl.com/win", "click-here.io", "free-gift.co",
              "prize-now.net", "offers.sale", "claim-now.me", "win-big.com"]

for i in range(320):
    t = random.choice(spam_templates).format(random.choice(spam_nouns), random.choice(spam_links))
    s.append(("spam", t))

df = pd.DataFrame(h + s, columns=["label", "message"])
df = df.sample(frac=1, random_state=1).reset_index(drop=True)

os.makedirs("data", exist_ok=True)
df.to_csv("data/messages.csv", index=False)
print(f"Generated {len(df)} messages ({sum(df['label']=='spam')} spam, {sum(df['label']=='ham')} ham)")
