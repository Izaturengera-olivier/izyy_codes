
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import axios from "axios";
import { ArrowRightIcon } from "@heroicons/react/24/solid";

const Home = () => {
  const [posts, setPosts] = useState([]);

  useEffect(() => {
    axios.get("http://127.0.0.1:8000/posts/")
      .then(response => {
        console.log("API Response:", response.data);
        setPosts(response.data);
      })
      .catch(error => console.error("Error fetching posts:", error));
  }, []);

  return (
    <div className="bg-gray-100 min-h-screen font-sans">
      <header className="bg-gray-800 text-white py-20 text-center shadow-md px-6 md:px-12">
        <h1 className="text-4xl font-bold">Welcome to Our Blog</h1>
        <p className="mt-3 text-lg">Explore insightful articles and stay informed</p>
      </header>

      <section className="container mx-auto px-6 py-12">
        <h2 className="text-3xl font-semibold text-gray-800 mb-6 text-center">Latest Posts</h2>
        <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-8">
          {posts.length > 0 ? (
            posts.map((post) => (
              <div key={post.id} className="bg-white rounded-lg shadow-md overflow-hidden">
                {post.featured_image && (
                  <img src={post.featured_image} alt={post.title} className="w-full h-48 object-cover" />
                )}

                <div className="p-4">
                  <h3 className="text-xl font-bold text-gray-900">{post.title}</h3>
                  <p className="text-gray-600 text-sm mt-2">By {post.author}</p>
                  <Link to={`/post/${post.slug}`} className="text-blue-500 flex items-center mt-3 hover:underline font-medium">
                    Read More <ArrowRightIcon className="w-4 h-4 ml-2" />
                  </Link>
                </div>
              </div>
            ))
          ) : (
            <p className="text-center text-gray-600">No posts available</p>
          )}
        </div>
      </section>
    </div>
  );
};

export default Home;
